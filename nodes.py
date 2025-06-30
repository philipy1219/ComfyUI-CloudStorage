"""
ComfyUI Cloud Storage Node
This module provides nodes for cloud storage operations in ComfyUI.
It supports multiple cloud storage providers including Aliyun OSS and AWS S3.
"""

import io
import json
import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo
from datetime import datetime
from utils import imageOrLatent, floatOrInt, BIGMAX, DIMMAX, get_load_formats, load_video
from comfy.cli_args import args
import os
import json

def get_cloud_credentials(storage_type):
    """
    从环境变量获取云存储凭证
    
    Args:
        storage_type (str): 存储类型 ('oss' 或 's3')
    
    Returns:
        tuple: (access_key_id, access_key_secret)
    """
    if storage_type == "oss":
        access_key_id = os.environ.get('ALIYUN_ACCESS_KEY_ID')
        access_key_secret = os.environ.get('ALIYUN_ACCESS_KEY_SECRET')
    else:  # s3
        access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
        access_key_secret = os.environ.get('AWS_SECRET_ACCESS_KEY')
    
    if not access_key_id or not access_key_secret:
        raise ValueError(f"Missing required environment variables for {storage_type} storage")
    
    return access_key_id, access_key_secret

class CloudStorageConfig:
    """
    Base configuration class for cloud storage operations.
    Provides unified interface for different cloud storage providers.
    """
    STORAGE_TYPES = ["oss", "s3"]
    
    @staticmethod
    def create_client(storage_type, **kwargs):
        """
        Create a cloud storage client based on the storage type.
        
        Args:
            storage_type (str): Type of storage service ('oss' or 's3')
            **kwargs: Configuration parameters including:
                - endpoint: Service endpoint URL
                - bucket: Bucket name
                - region: Region name (for S3)
        
        Returns:
            object: Cloud storage client (oss2.Bucket for OSS, boto3.client for S3)
        
        Raises:
            ImportError: If required package is not installed
            ValueError: If storage type is not supported
        """
        access_key_id, access_key_secret = get_cloud_credentials(storage_type)
        
        if storage_type == "oss":
            try:
                import oss2
                auth = oss2.Auth(access_key_id, access_key_secret)
                return oss2.Bucket(auth, kwargs.get('endpoint'), kwargs.get('bucket'))
            except ImportError:
                raise ImportError("Please install oss2 first: pip install oss2")
        elif storage_type == "s3":
            try:
                import boto3
                return boto3.client('s3',
                    aws_access_key_id=access_key_id,
                    aws_secret_access_key=access_key_secret,
                    endpoint_url=kwargs.get('endpoint'),
                    region_name=kwargs.get('region', 'us-east-1')
                )
            except ImportError:
                raise ImportError("Please install boto3 first: pip install boto3")
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")


class SaveImageToCloud:
    """
    Node for saving images to cloud storage.
    Supports both Aliyun OSS and AWS S3.
    """
    def __init__(self):
        """Initialize with output type and compression settings"""
        self.type = "output"
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        """
        Define input parameters for the node.
        Returns dict with required and optional parameters.
        """
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Images to be saved"}),
                "storage_type": (CloudStorageConfig.STORAGE_TYPES, {"default": "oss"}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "Prefix for the filename"}),
                "endpoint": ("STRING", {"default": ""}),
                "bucket": ("STRING", {"default": ""}),
                "prefix": ("STRING", {"default": "comfyui/", "tooltip": "Storage path prefix"}),
            },
            "optional": {
                "region": ("STRING", {"default": "us-east-1", "tooltip": "Region for S3 storage only"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("urls",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "cloud_storage"
    DESCRIPTION = "Save images to cloud storage (supports Aliyun OSS and AWS S3)"

    def save_images(self, images, storage_type, filename_prefix, endpoint, bucket,
                   prefix, region="us-east-1", prompt=None, extra_pnginfo=None):
        """
        Save images to cloud storage and return their URLs.
        
        Args:
            images: Image tensor to save
            storage_type: Type of cloud storage ('oss' or 's3')
            filename_prefix: Prefix for generated filenames
            endpoint: Storage service endpoint
            bucket: Storage bucket name
            prefix: Path prefix in storage
            region: Region for S3 (optional)
            prompt: Image generation prompt (optional)
            extra_pnginfo: Additional PNG metadata (optional)
        
        Returns:
            tuple: Contains semicolon-separated list of image URLs
        """
        client = CloudStorageConfig.create_client(
            storage_type,
            endpoint=endpoint,
            bucket=bucket,
            region=region
        )
        
        results = []
        for batch_number, image in enumerate(images):
            # Convert image format
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}{filename_prefix}_{timestamp}_{batch_number}.png"
            
            # Save image to memory
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG', pnginfo=metadata, compress_level=self.compress_level)
            img_byte_arr.seek(0)
            
            # Upload to cloud storage
            if storage_type == "oss":
                client.put_object(filename, img_byte_arr.getvalue())
                url = f"https://{bucket}.{endpoint}/{filename}"
            else:  # s3
                client.upload_fileobj(img_byte_arr, bucket, filename)
                if endpoint:
                    url = f"{endpoint}/{bucket}/{filename}"
                else:
                    url = f"https://{bucket}.s3.{region}.amazonaws.com/{filename}"
            
            results.append(url)
        
        return (";".join(results),)

class LoadImageFromCloud:
    """
    Node for loading images from cloud storage.
    Supports both Aliyun OSS and AWS S3.
    """
    @classmethod
    def INPUT_TYPES(s):
        """Define input parameters for the node"""
        return {
            "required": {
                "storage_type": (CloudStorageConfig.STORAGE_TYPES, {"default": "oss"}),
                "endpoint": ("STRING", {"default": ""}),
                "bucket": ("STRING", {"default": ""}),
                "file_key": ("STRING", {"default": "", "tooltip": "File path in cloud storage"}),
            },
            "optional": {
                "region": ("STRING", {"default": "us-east-1", "tooltip": "Region for S3 storage only"}),
            }
        }

    CATEGORY = "cloud_storage"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    DESCRIPTION = "Load images from cloud storage (supports Aliyun OSS and AWS S3)"

    def load_image(self, storage_type, endpoint, bucket, file_key, region="us-east-1"):
        """
        Load image from cloud storage and return image tensor with mask.
        
        Args:
            storage_type: Type of cloud storage ('oss' or 's3')
            endpoint: Storage service endpoint
            bucket: Storage bucket name
            file_key: Path to file in storage
            region: Region for S3 (optional)
        
        Returns:
            tuple: (image_tensor, mask_tensor)
        """
        client = CloudStorageConfig.create_client(
            storage_type,
            endpoint=endpoint,
            bucket=bucket,
            region=region
        )

        # Download file from cloud storage
        img_byte_arr = io.BytesIO()
        try:
            if storage_type == "oss":
                img_byte_arr.write(client.get_object(file_key).read())
            else:  # s3
                client.download_fileobj(bucket, file_key, img_byte_arr)
        except Exception as e:
            raise ValueError(f"Failed to load image from cloud storage: {str(e)}")
        
        img_byte_arr.seek(0)
        try:
            i = Image.open(img_byte_arr)
            i = ImageOps.exif_transpose(i)
        except Exception as e:
            raise ValueError(f"Failed to open image: {str(e)}")

        # Process image and create mask
        output_images = []
        output_masks = []
        w, h = None, None
        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            
            if len(output_images) == 0:
                w, h = image.size
            
            if image.size != (w, h):
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]

            # Process mask
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

class LoadMaskFromCloud:
    """
    Node for loading image masks from cloud storage.
    Supports extracting masks from specific color channels.
    """
    _color_channels = ["alpha", "red", "green", "blue"]
    
    @classmethod
    def INPUT_TYPES(s):
        """Define input parameters for the node"""
        return {
            "required": {
                "storage_type": (CloudStorageConfig.STORAGE_TYPES, {"default": "oss"}),
                "endpoint": ("STRING", {"default": ""}),
                "bucket": ("STRING", {"default": ""}),
                "file_key": ("STRING", {"default": "", "tooltip": "File path in cloud storage"}),
                "channel": (s._color_channels, {"default": "alpha", "tooltip": "Color channel to use as mask"}),
            },
            "optional": {
                "region": ("STRING", {"default": "us-east-1", "tooltip": "Region for S3 storage only"}),
            }
        }

    CATEGORY = "cloud_storage"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "load_mask"
    DESCRIPTION = "Load image mask from cloud storage (supports Aliyun OSS and AWS S3)"

    def load_mask(self, storage_type, endpoint, bucket, file_key, channel, region="us-east-1"):
        """
        Load image from cloud storage and extract mask from specified channel.
        
        Args:
            storage_type: Type of cloud storage ('oss' or 's3')
            endpoint: Storage service endpoint
            bucket: Storage bucket name
            file_key: Path to file in storage
            channel: Color channel to use as mask
            region: Region for S3 (optional)
        
        Returns:
            tuple: Contains single mask tensor
        """
        client = CloudStorageConfig.create_client(
            storage_type,
            endpoint=endpoint,
            bucket=bucket,
            region=region
        )

        # Download file from cloud storage
        img_byte_arr = io.BytesIO()
        try:
            if storage_type == "oss":
                img_byte_arr.write(client.get_object(file_key).read())
            else:  # s3
                client.download_fileobj(bucket, file_key, img_byte_arr)
        except Exception as e:
            raise ValueError(f"Failed to load image from cloud storage: {str(e)}")
        
        img_byte_arr.seek(0)
        try:
            i = Image.open(img_byte_arr)
            i = ImageOps.exif_transpose(i)
        except Exception as e:
            raise ValueError(f"Failed to open image: {str(e)}")

        # Ensure correct color channels
        if i.getbands() != ("R", "G", "B", "A"):
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            i = i.convert("RGBA")

        # Get mask from specified channel
        mask = None
        c = channel[0].upper()
        if c in i.getbands():
            mask = np.array(i.getchannel(c)).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask)
            if c == 'A':
                mask = 1. - mask
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")

        return (mask.unsqueeze(0),)

    @classmethod
    def IS_CHANGED(s, storage_type, endpoint, bucket, access_key_id, 
                   access_key_secret, file_key, channel, region="us-east-1"):
        """
        Check if the input parameters have changed.
        For cloud storage, we assume content might change at any time.
        """
        return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(s, storage_type, endpoint, bucket, file_key, channel, region="us-east-1"):
        """
        Validate input parameters before processing.
        
        Returns:
            bool or str: True if valid, error message if invalid
        """
        if not file_key:
            return "File path cannot be empty"
        if not endpoint or not bucket:
            return "Storage configuration is incomplete"
        return True

class LoadVideoFromCloud:
    """
    Node for loading videos from cloud storage.
    Supports both Aliyun OSS and AWS S3, with video processing capabilities.
    """
    @classmethod
    def INPUT_TYPES(s):
        """
        Define input parameters for video loading from cloud storage.
        """
        return {
            "required": {
                "storage_type": (CloudStorageConfig.STORAGE_TYPES, {"default": "oss"}),
                "endpoint": ("STRING", {"default": ""}),
                "bucket": ("STRING", {"default": ""}),
                "file_key": ("STRING", {"default": "", "tooltip": "Video file path in cloud storage"}),
                "force_rate": (floatOrInt, {"default": 0, "min": 0, "max": 60, "step": 1, "disable": 0}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": DIMMAX, 'disable': 0}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": DIMMAX, 'disable': 0}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1, "disable": 0}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
            },
            "optional": {
                "region": ("STRING", {"default": "us-east-1", "tooltip": "Region for S3 storage only"}),
                "meta_batch": ("VHS_BatchManager",),
                "vae": ("VAE",),
                "format": get_load_formats(),
            },
            "hidden": {
                "force_size": "STRING",
                "unique_id": "UNIQUE_ID"
            },
        }

    CATEGORY = "cloud_storage"
    RETURN_TYPES = (imageOrLatent, "INT", "AUDIO", "VHS_VIDEOINFO")
    RETURN_NAMES = ("IMAGE", "frame_count", "audio", "video_info")
    FUNCTION = "load_video"
    DESCRIPTION = "Load video from cloud storage (supports Aliyun OSS and AWS S3)"

    def load_video(self, storage_type, endpoint, bucket, file_key, force_rate=0,
                  custom_width=0, custom_height=0, frame_load_cap=0, skip_first_frames=0,
                  select_every_nth=1, region="us-east-1", meta_batch=None, vae=None,
                  format=None, force_size=None, unique_id=None):
        """
        Load and process video from cloud storage.
        """
        try:
            # Create cloud storage client
            client = CloudStorageConfig.create_client(
                storage_type,
                endpoint=endpoint,
                bucket=bucket,
                region=region
            )

            # Download video to temporary file
            import tempfile
            import os
            from pathlib import Path
            
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"comfyui_video_{Path(file_key).name}")
            
            try:
                if storage_type == "oss":
                    client.get_object_to_file(file_key, temp_path)
                else:  # s3
                    client.download_file(bucket, file_key, temp_path)
                
                # Create kwargs for load_video function
                video_kwargs = {
                    "video": temp_path,
                    "force_rate": force_rate,
                    "custom_width": custom_width,
                    "custom_height": custom_height,
                    "frame_load_cap": frame_load_cap,
                    "skip_first_frames": skip_first_frames,
                    "select_every_nth": select_every_nth,
                    "meta_batch": meta_batch,
                    "vae": vae,
                    "format": format,
                    "force_size": force_size,
                    "unique_id": unique_id
                }
                
                # Use existing load_video function
                return load_video(**video_kwargs)
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except Exception as e:
            raise ValueError(f"Failed to load video from cloud storage: {str(e)}")

    @classmethod
    def IS_CHANGED(s, storage_type, endpoint, bucket, access_key_id, 
                   access_key_secret, file_key, **kwargs):
        """Always return different hash to ensure reloading"""
        return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(s, storage_type, endpoint, bucket, file_key, **kwargs):
        """Validate input parameters before processing."""
        if not file_key:
            return "File path cannot be empty"
        if not endpoint or not bucket:
            return "Storage configuration is incomplete"
        return True

class UploadFileToCloud:
    """
    通用节点，用于将任意文件上传到云存储。
    支持阿里云OSS和AWS S3。
    """
    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        """定义节点的输入参数"""
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "tooltip": "要上传的本地文件路径"}),
                "storage_type": (CloudStorageConfig.STORAGE_TYPES, {"default": "oss"}),
                "endpoint": ("STRING", {"default": ""}),
                "bucket": ("STRING", {"default": ""}),
                "prefix": ("STRING", {"default": "uploads/", "tooltip": "存储路径前缀"}),
            },
            "optional": {
                "region": ("STRING", {"default": "us-east-1", "tooltip": "S3存储区域"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("url",)
    FUNCTION = "upload_file"
    OUTPUT_NODE = True
    CATEGORY = "cloud_storage"
    DESCRIPTION = "上传任意文件到云存储 (支持阿里云OSS和AWS S3)"

    def upload_file(self, file_path, storage_type, endpoint, bucket, prefix, region="us-east-1"):
        """
        上传文件到云存储并返回URL。
        
        Args:
            file_path: 本地文件路径
            storage_type: 存储类型 ('oss' 或 's3')
            endpoint: 服务端点URL
            bucket: 存储桶名称
            prefix: 存储路径前缀
            region: S3区域 (可选)
        
        Returns:
            tuple: 包含文件URL的元组
        """
        if not os.path.exists(file_path):
            raise ValueError(f"文件不存在: {file_path}")

        client = CloudStorageConfig.create_client(
            storage_type,
            endpoint=endpoint,
            bucket=bucket,
            region=region
        )

        # 生成云存储文件名
        filename = os.path.basename(file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cloud_filename = f"{prefix}{timestamp}_{filename}"

        try:
            # 上传文件到云存储
            if storage_type == "oss":
                client.put_object_from_file(cloud_filename, file_path)
                url = f"https://{bucket}.{endpoint}/{cloud_filename}"
            else:  # s3
                with open(file_path, 'rb') as f:
                    client.upload_fileobj(f, bucket, cloud_filename)
                if endpoint:
                    url = f"{endpoint}/{bucket}/{cloud_filename}"
                else:
                    url = f"https://{bucket}.s3.{region}.amazonaws.com/{cloud_filename}"
            
            return (url,)
        except Exception as e:
            raise ValueError(f"上传文件失败: {str(e)}")

    @classmethod
    def VALIDATE_INPUTS(s, file_path, storage_type, endpoint, bucket, prefix, region="us-east-1"):
        """验证输入参数"""
        if not file_path:
            return "文件路径不能为空"
        if not os.path.exists(file_path):
            return "文件不存在"
        if not endpoint or not bucket:
            return "存储配置不完整"
        return True

# Register nodes
NODE_CLASS_MAPPINGS = {
    "SaveImageToCloud": SaveImageToCloud,
    "LoadImageFromCloud": LoadImageFromCloud,
    "LoadMaskFromCloud": LoadMaskFromCloud,
    "LoadVideoFromCloud": LoadVideoFromCloud,
    "UploadFileToCloud": UploadFileToCloud,
}

# Node display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageToCloud": "Save Image To Cloud",
    "LoadImageFromCloud": "Load Image From Cloud",
    "LoadMaskFromCloud": "Load Mask From Cloud",
    "LoadVideoFromCloud": "Load Video From Cloud",
    "UploadFileToCloud": "Upload File To Cloud",
}