"""
ComfyUI Cloud Storage Node
This module provides nodes for cloud storage operations in ComfyUI.
It supports multiple cloud storage providers including Aliyun OSS and AWS S3.
"""

import sys
import io
import json
import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence, ExifTags
from PIL.PngImagePlugin import PngInfo
from datetime import datetime
from utils import imageOrLatent, floatOrInt, BIGMAX, DIMMAX, get_load_formats, load_video, cached, gifski_path, ContainsAll, tensor_to_bytes, logger, tensor_to_shorts, to_pingpong, ffmpeg_path, ENCODE_ARGS, merge_filter_args
from comfy.cli_args import args
import folder_paths
import os
import json
import itertools
import re
import subprocess
from comfy.utils import ProgressBar
from string import Template

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

def flatten_list(l):
    ret = []
    for e in l:
        if isinstance(e, list):
            ret.extend(e)
        else:
            ret.append(e)
    return ret

def iterate_format(video_format, for_widgets=True):
    """Provides an iterator over widgets, or arguments"""
    def indirector(cont, index):
        if isinstance(cont[index], list) and (not for_widgets
          or len(cont[index])> 1 and not isinstance(cont[index][1], dict)):
            inp = yield cont[index]
            if inp is not None:
                cont[index] = inp
                yield
    for k in video_format:
        if k == "extra_widgets":
            if for_widgets:
                yield from video_format["extra_widgets"]
        elif k.endswith("_pass"):
            for i in range(len(video_format[k])):
                yield from indirector(video_format[k], i)
            if not for_widgets:
                video_format[k] = flatten_list(video_format[k])
        else:
            yield from indirector(video_format, k)

base_formats_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "video_formats")
@cached(5)
def get_video_formats():
    format_files = {}
    for format_name in folder_paths.get_filename_list("VHS_video_formats"):
        format_files[format_name] = folder_paths.get_full_path("VHS_video_formats", format_name)
    for item in os.scandir(base_formats_dir):
        if not item.is_file() or not item.name.endswith('.json'):
            continue
        format_files[item.name[:-5]] = item.path
    formats = []
    format_widgets = {}
    for format_name, path in format_files.items():
        with open(path, 'r') as stream:
            video_format = json.load(stream)
        if "gifski_pass" in video_format and gifski_path is None:
            #Skip format
            continue
        widgets = list(iterate_format(video_format))
        formats.append("video/" + format_name)
        if (len(widgets) > 0):
            format_widgets["video/"+ format_name] = widgets
    return formats, format_widgets

def apply_format_widgets(format_name, kwargs):
    if os.path.exists(os.path.join(base_formats_dir, format_name + ".json")):
        video_format_path = os.path.join(base_formats_dir, format_name + ".json")
    else:
        video_format_path = folder_paths.get_full_path("VHS_video_formats", format_name)
    with open(video_format_path, 'r') as stream:
        video_format = json.load(stream)
    for w in iterate_format(video_format):
        if w[0] not in kwargs:
            if len(w) > 2 and 'default' in w[2]:
                default = w[2]['default']
            else:
                if type(w[1]) is list:
                    default = w[1][0]
                else:
                    #NOTE: This doesn't respect max/min, but should be good enough as a fallback to a fallback to a fallback
                    default = {"BOOLEAN": False, "INT": 0, "FLOAT": 0, "STRING": ""}[w[1]]
            kwargs[w[0]] = default
            logger.warn(f"Missing input for {w[0][0]} has been set to {default}")
    wit = iterate_format(video_format, False)
    for w in wit:
        while isinstance(w, list):
            if len(w) == 1:
                #TODO: mapping=kwargs should be safer, but results in key errors, investigate why
                w = [Template(x).substitute(**kwargs) for x in w[0]]
                break
            elif isinstance(w[1], dict):
                w = w[1][str(kwargs[w[0]])]
            elif len(w) > 3:
                w = Template(w[3]).substitute(val=kwargs[w[0]])
            else:
                w = str(kwargs[w[0]])
        wit.send(w)
    return video_format

def ffmpeg_process(args, video_format, video_metadata, file_path, env):

    res = None
    frame_data = yield
    total_frames_output = 0
    if video_format.get('save_metadata', 'False') != 'False':
        os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
        metadata = json.dumps(video_metadata)
        metadata_path = os.path.join(folder_paths.get_temp_directory(), "metadata.txt")
        #metadata from file should  escape = ; # \ and newline
        metadata = metadata.replace("\\","\\\\")
        metadata = metadata.replace(";","\\;")
        metadata = metadata.replace("#","\\#")
        metadata = metadata.replace("=","\\=")
        metadata = metadata.replace("\n","\\\n")
        metadata = "comment=" + metadata
        with open(metadata_path, "w") as f:
            f.write(";FFMETADATA1\n")
            f.write(metadata)
        m_args = args[:1] + ["-i", metadata_path] + args[1:] + ["-metadata", "creation_time=now"]
        with subprocess.Popen(m_args + [file_path], stderr=subprocess.PIPE,
                              stdin=subprocess.PIPE, env=env) as proc:
            try:
                while frame_data is not None:
                    proc.stdin.write(frame_data)
                    #TODO: skip flush for increased speed
                    frame_data = yield
                    total_frames_output+=1
                proc.stdin.flush()
                proc.stdin.close()
                res = proc.stderr.read()
            except BrokenPipeError as e:
                err = proc.stderr.read()
                #Check if output file exists. If it does, the re-execution
                #will also fail. This obscures the cause of the error
                #and seems to never occur concurrent to the metadata issue
                if os.path.exists(file_path):
                    raise Exception("An error occurred in the ffmpeg subprocess:\n" \
                            + err.decode(*ENCODE_ARGS))
                #Res was not set
                print(err.decode(*ENCODE_ARGS), end="", file=sys.stderr)
                logger.warn("An error occurred when saving with metadata")
    if res != b'':
        with subprocess.Popen(args + [file_path], stderr=subprocess.PIPE,
                              stdin=subprocess.PIPE, env=env) as proc:
            try:
                while frame_data is not None:
                    proc.stdin.write(frame_data)
                    frame_data = yield
                    total_frames_output+=1
                proc.stdin.flush()
                proc.stdin.close()
                res = proc.stderr.read()
            except BrokenPipeError as e:
                res = proc.stderr.read()
                raise Exception("An error occurred in the ffmpeg subprocess:\n" \
                        + res.decode(*ENCODE_ARGS))
    yield total_frames_output
    if len(res) > 0:
        print(res.decode(*ENCODE_ARGS), end="", file=sys.stderr)

def gifski_process(args, dimensions, video_format, file_path, env):
    frame_data = yield
    with subprocess.Popen(args + video_format['main_pass'] + ['-f', 'yuv4mpegpipe', '-'],
                          stderr=subprocess.PIPE, stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, env=env) as procff:
        with subprocess.Popen([gifski_path] + video_format['gifski_pass']
                              + ['-W', f'{dimensions[0]}', '-H', f'{dimensions[1]}']
                              + ['-q', '-o', file_path, '-'], stderr=subprocess.PIPE,
                              stdin=procff.stdout, stdout=subprocess.PIPE,
                              env=env) as procgs:
            try:
                while frame_data is not None:
                    procff.stdin.write(frame_data)
                    frame_data = yield
                procff.stdin.flush()
                procff.stdin.close()
                resff = procff.stderr.read()
                resgs = procgs.stderr.read()
                outgs = procgs.stdout.read()
            except BrokenPipeError as e:
                procff.stdin.close()
                resff = procff.stderr.read()
                resgs = procgs.stderr.read()
                raise Exception("An error occurred while creating gifski output\n" \
                        + "Make sure you are using gifski --version >=1.32.0\nffmpeg: " \
                        + resff.decode(*ENCODE_ARGS) + '\ngifski: ' + resgs.decode(*ENCODE_ARGS))
    if len(resff) > 0:
        print(resff.decode(*ENCODE_ARGS), end="", file=sys.stderr)
    if len(resgs) > 0:
        print(resgs.decode(*ENCODE_ARGS), end="", file=sys.stderr)
    #should always be empty as the quiet flag is passed
    if len(outgs) > 0:
        print(outgs.decode(*ENCODE_ARGS))

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

class VideoCombineToCloud:
    @classmethod
    def INPUT_TYPES(s):
        ffmpeg_formats, format_widgets = get_video_formats()
        format_widgets["image/webp"] = [['lossless', "BOOLEAN", {'default': True}]]
        return {
            "required": {
                "images": (imageOrLatent,),
                "frame_rate": (
                    floatOrInt,
                    {"default": 8, "min": 1, "step": 1},
                ),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "filename_prefix": ("STRING", {"default": "AnimateDiff"}),
                "format": (["image/gif", "image/webp"] + ffmpeg_formats, {'formats': format_widgets}),
                "pingpong": ("BOOLEAN", {"default": False}),
                "save_output": ("BOOLEAN", {"default": True}),
                "save_to": (["local", "cloud"], {"default": "local"}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "meta_batch": ("VHS_BatchManager",),
                "vae": ("VAE",),
                # Cloud storage parameters
                "storage_type": (CloudStorageConfig.STORAGE_TYPES, {"default": "oss"}),
                "endpoint": ("STRING", {"default": ""}),
                "bucket": ("STRING", {"default": ""}),
                "prefix": ("STRING", {"default": "comfyui/", "tooltip": "Storage path prefix"}),
                "region": ("STRING", {"default": "us-east-1", "tooltip": "Region for S3 storage only"}),
            },
            "hidden": ContainsAll({
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            }),
        }

    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("Filenames",)
    OUTPUT_NODE = True
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "combine_video"

    def combine_video(self, frame_rate, loop_count, images=None, latents=None,
                     filename_prefix="AnimateDiff", format="image/gif", pingpong=False,
                     save_output=True, save_to="local", prompt=None, extra_pnginfo=None,
                     audio=None, unique_id=None, manual_format_widgets=None,
                     meta_batch=None, vae=None, storage_type=None, endpoint=None,
                     bucket=None, prefix="comfyui/", region="us-east-1", **kwargs):
        if latents is not None:
            images = latents
        if images is None:
            return ((save_output, []),)
        if vae is not None:
            if isinstance(images, dict):
                images = images['samples']
            else:
                vae = None

        if isinstance(images, torch.Tensor) and images.size(0) == 0:
            return ((save_output, []),)
        num_frames = len(images)
        pbar = ProgressBar(num_frames)
        if vae is not None:
            downscale_ratio = getattr(vae, "downscale_ratio", 8)
            width = images.size(-1)*downscale_ratio
            height = images.size(-2)*downscale_ratio
            frames_per_batch = (1920 * 1080 * 16) // (width * height) or 1
            #Python 3.12 adds an itertools.batched, but it's easily replicated for legacy support
            def batched(it, n):
                while batch := tuple(itertools.islice(it, n)):
                    yield batch
            def batched_encode(images, vae, frames_per_batch):
                for batch in batched(iter(images), frames_per_batch):
                    image_batch = torch.from_numpy(np.array(batch))
                    yield from vae.decode(image_batch)
            images = batched_encode(images, vae, frames_per_batch)
            first_image = next(images)
            #repush first_image
            images = itertools.chain([first_image], images)
            #A single image has 3 dimensions. Discard higher dimensions
            while len(first_image.shape) > 3:
                first_image = first_image[0]
        else:
            first_image = images[0]
            images = iter(images)
        # get output information
        output_dir = (
            folder_paths.get_output_directory()
            if save_output
            else folder_paths.get_temp_directory()
        )
        (
            full_output_folder,
            filename,
            _,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path(filename_prefix, output_dir)
        output_files = []

        metadata = PngInfo()
        video_metadata = {}
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
            video_metadata["prompt"] = json.dumps(prompt)
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                video_metadata[x] = extra_pnginfo[x]
            extra_options = extra_pnginfo.get('workflow', {}).get('extra', {})
        else:
            extra_options = {}
        metadata.add_text("CreationTime", datetime.datetime.now().isoformat(" ")[:19])

        if meta_batch is not None and unique_id in meta_batch.outputs:
            (counter, output_process) = meta_batch.outputs[unique_id]
        else:
            # comfy counter workaround
            max_counter = 0

            # Loop through the existing files
            matcher = re.compile(f"{re.escape(filename)}_(\\d+)\\D*\\..+", re.IGNORECASE)
            for existing_file in os.listdir(full_output_folder):
                # Check if the file matches the expected format
                match = matcher.fullmatch(existing_file)
                if match:
                    # Extract the numeric portion of the filename
                    file_counter = int(match.group(1))
                    # Update the maximum counter value if necessary
                    if file_counter > max_counter:
                        max_counter = file_counter

            # Increment the counter by 1 to get the next available value
            counter = max_counter + 1
            output_process = None

        # save first frame as png to keep metadata
        first_image_file = f"{filename}_{counter:05}.png"
        file_path = os.path.join(full_output_folder, first_image_file)
        if extra_options.get('VHS_MetadataImage', True) != False:
            Image.fromarray(tensor_to_bytes(first_image)).save(
                file_path,
                pnginfo=metadata,
                compress_level=4,
            )
        output_files.append(file_path)

        format_type, format_ext = format.split("/")
        if format_type == "image":
            if meta_batch is not None:
                raise Exception("Pillow('image/') formats are not compatible with batched output")
            image_kwargs = {}
            if format_ext == "gif":
                image_kwargs['disposal'] = 2
            if format_ext == "webp":
                #Save timestamp information
                exif = Image.Exif()
                exif[ExifTags.IFD.Exif] = {36867: datetime.datetime.now().isoformat(" ")[:19]}
                image_kwargs['exif'] = exif
                image_kwargs['lossless'] = kwargs.get("lossless", True)
            file = f"{filename}_{counter:05}.{format_ext}"
            file_path = os.path.join(full_output_folder, file)
            if pingpong:
                images = to_pingpong(images)
            def frames_gen(images):
                for i in images:
                    pbar.update(1)
                    yield Image.fromarray(tensor_to_bytes(i))
            frames = frames_gen(images)
            # Use pillow directly to save an animated image
            next(frames).save(
                file_path,
                format=format_ext.upper(),
                save_all=True,
                append_images=frames,
                duration=round(1000 / frame_rate),
                loop=loop_count,
                compress_level=4,
                **image_kwargs
            )
            output_files.append(file_path)
        else:
            # Use ffmpeg to save a video
            if ffmpeg_path is None:
                raise ProcessLookupError(f"ffmpeg is required for video outputs and could not be found.\nIn order to use video outputs, you must either:\n- Install imageio-ffmpeg with pip,\n- Place a ffmpeg executable in {os.path.abspath('')}, or\n- Install ffmpeg and add it to the system path.")

            if manual_format_widgets is not None:
                logger.warn("Format args can now be passed directly. The manual_format_widgets argument is now deprecated")
                kwargs.update(manual_format_widgets)

            has_alpha = first_image.shape[-1] == 4
            kwargs["has_alpha"] = has_alpha
            video_format = apply_format_widgets(format_ext, kwargs)
            dim_alignment = video_format.get("dim_alignment", 2)
            if (first_image.shape[1] % dim_alignment) or (first_image.shape[0] % dim_alignment):
                #output frames must be padded
                to_pad = (-first_image.shape[1] % dim_alignment,
                          -first_image.shape[0] % dim_alignment)
                padding = (to_pad[0]//2, to_pad[0] - to_pad[0]//2,
                           to_pad[1]//2, to_pad[1] - to_pad[1]//2)
                padfunc = torch.nn.ReplicationPad2d(padding)
                def pad(image):
                    image = image.permute((2,0,1))#HWC to CHW
                    padded = padfunc(image.to(dtype=torch.float32))
                    return padded.permute((1,2,0))
                images = map(pad, images)
                dimensions = (-first_image.shape[1] % dim_alignment + first_image.shape[1],
                              -first_image.shape[0] % dim_alignment + first_image.shape[0])
                logger.warn("Output images were not of valid resolution and have had padding applied")
            else:
                dimensions = (first_image.shape[1], first_image.shape[0])
            if loop_count > 0:
                loop_args = ["-vf", "loop=loop=" + str(loop_count)+":size=" + str(num_frames)]
            else:
                loop_args = []
            if pingpong:
                if meta_batch is not None:
                    logger.error("pingpong is incompatible with batched output")
                images = to_pingpong(images)
            if video_format.get('input_color_depth', '8bit') == '16bit':
                images = map(tensor_to_shorts, images)
                if has_alpha:
                    i_pix_fmt = 'rgba64'
                else:
                    i_pix_fmt = 'rgb48'
            else:
                images = map(tensor_to_bytes, images)
                if has_alpha:
                    i_pix_fmt = 'rgba'
                else:
                    i_pix_fmt = 'rgb24'
            file = f"{filename}_{counter:05}.{video_format['extension']}"
            file_path = os.path.join(full_output_folder, file)
            bitrate_arg = []
            bitrate = video_format.get('bitrate')
            if bitrate is not None:
                bitrate_arg = ["-b:v", str(bitrate) + "M" if video_format.get('megabit') == 'True' else str(bitrate) + "K"]
            args = [ffmpeg_path, "-v", "error", "-f", "rawvideo", "-pix_fmt", i_pix_fmt,
                    # The image data is in an undefined generic RGB color space, which in practice means sRGB.
                    # sRGB has the same primaries and matrix as BT.709, but a different transfer function (gamma),
                    # called by the sRGB standard name IEC 61966-2-1. However, video hosting platforms like YouTube
                    # standardize on full BT.709 and will convert the colors accordingly. This last minute change
                    # in colors can be confusing to users. We can counter it by lying about the transfer function
                    # on a per format basis, i.e. for video we will lie to FFmpeg that it is already BT.709. Also,
                    # because the input data is in RGB (not YUV) it is more efficient (fewer scale filter invocations)
                    # to specify the input color space as RGB and then later, if the format actually wants YUV,
                    # to convert it to BT.709 YUV via FFmpeg's -vf "scale=out_color_matrix=bt709".
                    "-color_range", "pc", "-colorspace", "rgb", "-color_primaries", "bt709",
                    "-color_trc", video_format.get("fake_trc", "iec61966-2-1"),
                    "-s", f"{dimensions[0]}x{dimensions[1]}", "-r", str(frame_rate), "-i", "-"] \
                    + loop_args

            images = map(lambda x: x.tobytes(), images)
            env=os.environ.copy()
            if  "environment" in video_format:
                env.update(video_format["environment"])

            if "pre_pass" in video_format:
                if meta_batch is not None:
                    #Performing a prepass requires keeping access to all frames.
                    #Potential solutions include keeping just output frames in
                    #memory or using 3 passes with intermediate file, but
                    #very long gifs probably shouldn't be encouraged
                    raise Exception("Formats which require a pre_pass are incompatible with Batch Manager.")
                images = [b''.join(images)]
                os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
                in_args_len = args.index("-i") + 2 # The index after ["-i", "-"]
                pre_pass_args = args[:in_args_len] + video_format['pre_pass']
                merge_filter_args(pre_pass_args)
                try:
                    subprocess.run(pre_pass_args, input=images[0], env=env,
                                   capture_output=True, check=True)
                except subprocess.CalledProcessError as e:
                    raise Exception("An error occurred in the ffmpeg prepass:\n" \
                            + e.stderr.decode(*ENCODE_ARGS))
            if "inputs_main_pass" in video_format:
                in_args_len = args.index("-i") + 2 # The index after ["-i", "-"]
                args = args[:in_args_len] + video_format['inputs_main_pass'] + args[in_args_len:]

            if output_process is None:
                if 'gifski_pass' in video_format:
                    format = 'image/gif'
                    output_process = gifski_process(args, dimensions, video_format, file_path, env)
                else:
                    args += video_format['main_pass'] + bitrate_arg
                    merge_filter_args(args)
                    output_process = ffmpeg_process(args, video_format, video_metadata, file_path, env)
                #Proceed to first yield
                output_process.send(None)
                if meta_batch is not None:
                    meta_batch.outputs[unique_id] = (counter, output_process)

            for image in images:
                pbar.update(1)
                output_process.send(image)
            if meta_batch is None or meta_batch.has_closed_inputs:
                #Close pipe and wait for termination.
                try:
                    total_frames_output = output_process.send(None)
                    output_process.send(None)
                except StopIteration:
                    pass
                if meta_batch is not None:
                    meta_batch.outputs.pop(unique_id)
                    if len(meta_batch.outputs) == 0:
                        meta_batch.reset()
            else:
                #batch is unfinished
                #TODO: Check if empty output breaks other custom nodes
                return {"ui": {"unfinished_batch": [True]}, "result": ((save_output, []),)}

            output_files.append(file_path)


            a_waveform = None
            if audio is not None:
                try:
                    #safely check if audio produced by VHS_LoadVideo actually exists
                    a_waveform = audio['waveform']
                except:
                    pass
            if a_waveform is not None:
                # Create audio file if input was provided
                output_file_with_audio = f"{filename}_{counter:05}-audio.{video_format['extension']}"
                output_file_with_audio_path = os.path.join(full_output_folder, output_file_with_audio)
                if "audio_pass" not in video_format:
                    logger.warn("Selected video format does not have explicit audio support")
                    video_format["audio_pass"] = ["-c:a", "libopus"]


                # FFmpeg command with audio re-encoding
                #TODO: expose audio quality options if format widgets makes it in
                #Reconsider forcing apad/shortest
                channels = audio['waveform'].size(1)
                min_audio_dur = total_frames_output / frame_rate + 1
                if video_format.get('trim_to_audio', 'False') != 'False':
                    apad = []
                else:
                    apad = ["-af", "apad=whole_dur="+str(min_audio_dur)]
                mux_args = [ffmpeg_path, "-v", "error", "-n", "-i", file_path,
                            "-ar", str(audio['sample_rate']), "-ac", str(channels),
                            "-f", "f32le", "-i", "-", "-c:v", "copy"] \
                            + video_format["audio_pass"] \
                            + apad + ["-shortest", output_file_with_audio_path]

                audio_data = audio['waveform'].squeeze(0).transpose(0,1) \
                        .numpy().tobytes()
                merge_filter_args(mux_args, '-af')
                try:
                    res = subprocess.run(mux_args, input=audio_data,
                                         env=env, capture_output=True, check=True)
                except subprocess.CalledProcessError as e:
                    raise Exception("An error occured in the ffmpeg subprocess:\n" \
                            + e.stderr.decode(*ENCODE_ARGS))
                if res.stderr:
                    print(res.stderr.decode(*ENCODE_ARGS), end="", file=sys.stderr)
                output_files.append(output_file_with_audio_path)
                #Return this file with audio to the webui.
                #It will be muted unless opened or saved with right click
                file = output_file_with_audio
        if extra_options.get('VHS_KeepIntermediate', True) == False:
            for intermediate in output_files[1:-1]:
                if os.path.exists(intermediate):
                    os.remove(intermediate)
        preview = {
                "filename": file,
                "subfolder": subfolder,
                "type": "output" if save_output else "temp",
                "format": format,
                "frame_rate": frame_rate,
                "workflow": first_image_file,
                "fullpath": output_files[-1],
            }
        if num_frames == 1 and 'png' in format and '%03d' in file:
            preview['format'] = 'image/png'
            preview['filename'] = file.replace('%03d', '001')

        # After saving the video file, handle cloud storage if selected
        if save_to == "cloud" and save_output:
            if not all([storage_type, endpoint, bucket]):
                raise ValueError("Cloud storage configuration is incomplete")

            try:
                # Create cloud storage client
                client = CloudStorageConfig.create_client(
                    storage_type,
                    endpoint=endpoint,
                    bucket=bucket,
                    region=region
                )

                # Generate cloud filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cloud_filename = f"{prefix}video_{timestamp}_{filename_prefix}.{format_ext}"

                # Upload to cloud storage
                if storage_type == "oss":
                    client.put_object_from_file(cloud_filename, file_path)
                    url = f"https://{bucket}.{endpoint}/{cloud_filename}"
                else:  # s3
                    client.upload_file(file_path, bucket, cloud_filename)
                    if endpoint:
                        url = f"{endpoint}/{bucket}/{cloud_filename}"
                    else:
                        url = f"https://{bucket}.s3.{region}.amazonaws.com/{cloud_filename}"
                
                # Add cloud storage information to preview
                preview["cloud_url"] = url

            except Exception as e:
                raise ValueError(f"Failed to upload to cloud storage: {str(e)}")

        return {"ui": {"gifs": [preview]}, "result": ((save_output, output_files),)}

# Register nodes
NODE_CLASS_MAPPINGS = {
    "SaveImageToCloud": SaveImageToCloud,
    "LoadImageFromCloud": LoadImageFromCloud,
    "LoadMaskFromCloud": LoadMaskFromCloud,
    "LoadVideoFromCloud": LoadVideoFromCloud,
    "VideoCombineToCloud": VideoCombineToCloud
}

# Node display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageToCloud": "Save Image To Cloud",
    "LoadImageFromCloud": "Load Image From Cloud",
    "LoadMaskFromCloud": "Load Mask From Cloud",
    "LoadVideoFromCloud": "Load Video From Cloud",
    "VideoCombineToCloud": "Video Combine To Cloud"
}