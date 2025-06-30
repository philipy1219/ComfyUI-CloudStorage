import os
import shutil
import itertools
import numpy as np
import torch
import cv2
import psutil
import subprocess
import re
import time
from collections.abc import Mapping
from comfy.utils import common_upscale, ProgressBar
import nodes
import sys
import copy
import logging

class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[0;36m",  # CYAN
        "INFO": "\033[0;32m",  # GREEN
        "WARNING": "\033[0;33m",  # YELLOW
        "ERROR": "\033[0;31m",  # RED
        "CRITICAL": "\033[0;37;41m",  # WHITE ON RED
        "RESET": "\033[0m",  # RESET COLOR
    }

    def format(self, record):
        colored_record = copy.copy(record)
        levelname = colored_record.levelname
        seq = self.COLORS.get(levelname, self.COLORS["RESET"])
        colored_record.levelname = f"{seq}{levelname}{self.COLORS['RESET']}"
        return super().format(colored_record)


# Create a new logger
logger = logging.getLogger("ComfyUI Cloud Storage")
logger.propagate = False

# Add handler if we don't have one.
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter("[%(name)s] - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

# Configure logger
loglevel = logging.INFO
logger.setLevel(loglevel)

BIGMIN = -(2**53-1)
BIGMAX = (2**53-1)

DIMMAX = 8192

ENCODE_ARGS = ("utf-8", 'backslashreplace')

class MultiInput(str):
    def __new__(cls, string, allowed_types="*"):
        res = super().__new__(cls, string)
        res.allowed_types=allowed_types
        return res
    def __ne__(self, other):
        if self.allowed_types == "*" or other == "*":
            return False
        return other not in self.allowed_types
imageOrLatent = MultiInput("IMAGE", ["IMAGE", "LATENT"])
floatOrInt = MultiInput("FLOAT", ["FLOAT", "INT"])

def ffmpeg_suitability(path):
    try:
        version = subprocess.run([path, "-version"], check=True,
                                 capture_output=True).stdout.decode(*ENCODE_ARGS)
    except:
        return 0
    score = 0
    #rough layout of the importance of various features
    simple_criterion = [("libvpx", 20),("264",10), ("265",3),
                        ("svtav1",5),("libopus", 1)]
    for criterion in simple_criterion:
        if version.find(criterion[0]) >= 0:
            score += criterion[1]
    #obtain rough compile year from copyright information
    copyright_index = version.find('2000-2')
    if copyright_index >= 0:
        copyright_year = version[copyright_index+6:copyright_index+9]
        if copyright_year.isnumeric():
            score += int(copyright_year)
    return score

def get_audio(file, start_time=0, duration=0):
    args = [ffmpeg_path, "-i", file]
    if start_time > 0:
        args += ["-ss", str(start_time)]
    if duration > 0:
        args += ["-t", str(duration)]
    try:
        #TODO: scan for sample rate and maintain
        res =  subprocess.run(args + ["-f", "f32le", "-"],
                              capture_output=True, check=True)
        audio = torch.frombuffer(bytearray(res.stdout), dtype=torch.float32)
        match = re.search(', (\\d+) Hz, (\\w+), ',res.stderr.decode(*ENCODE_ARGS))
    except subprocess.CalledProcessError as e:
        raise Exception(f"VHS failed to extract audio from {file}:\n" \
                + e.stderr.decode(*ENCODE_ARGS))
    if match:
        ar = int(match.group(1))
        #NOTE: Just throwing an error for other channel types right now
        #Will deal with issues if they come
        ac = {"mono": 1, "stereo": 2}[match.group(2)]
    else:
        ar = 44100
        ac = 2
    audio = audio.reshape((-1,ac)).transpose(0,1).unsqueeze(0)
    return {'waveform': audio, 'sample_rate': ar}

class LazyAudioMap(Mapping):
    def __init__(self, file, start_time, duration):
        self.file = file
        self.start_time=start_time
        self.duration=duration
        self._dict=None
    def __getitem__(self, key):
        if self._dict is None:
            self._dict = get_audio(self.file, self.start_time, self.duration)
        return self._dict[key]
    def __iter__(self):
        if self._dict is None:
            self._dict = get_audio(self.file, self.start_time, self.duration)
        return iter(self._dict)
    def __len__(self):
        if self._dict is None:
            self._dict = get_audio(self.file, self.start_time, self.duration)
        return len(self._dict)

def lazy_get_audio(file, start_time=0, duration=0, **kwargs):
    return LazyAudioMap(file, start_time, duration)

def strip_path(path):
    #This leaves whitespace inside quotes and only a single "
    #thus ' ""test"' -> '"test'
    #consider path.strip(string.whitespace+"\"")
    #or weightier re.fullmatch("[\\s\"]*(.+?)[\\s\"]*", path).group(1)
    path = path.strip()
    if path.startswith("\""):
        path = path[1:]
    if path.endswith("\""):
        path = path[:-1]
    return path

if "VHS_FORCE_FFMPEG_PATH" in os.environ:
    ffmpeg_path = os.environ.get("VHS_FORCE_FFMPEG_PATH")
else:
    ffmpeg_paths = []
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        imageio_ffmpeg_path = get_ffmpeg_exe()
        ffmpeg_paths.append(imageio_ffmpeg_path)
    except:
        if "VHS_USE_IMAGEIO_FFMPEG" in os.environ:
            raise
        logger.warn("Failed to import imageio_ffmpeg")
    if "VHS_USE_IMAGEIO_FFMPEG" in os.environ:
        ffmpeg_path = imageio_ffmpeg_path
    else:
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is not None:
            ffmpeg_paths.append(system_ffmpeg)
        if os.path.isfile("ffmpeg"):
            ffmpeg_paths.append(os.path.abspath("ffmpeg"))
        if os.path.isfile("ffmpeg.exe"):
            ffmpeg_paths.append(os.path.abspath("ffmpeg.exe"))
        if len(ffmpeg_paths) == 0:
            logger.error("No valid ffmpeg found.")
            ffmpeg_path = None
        elif len(ffmpeg_paths) == 1:
            #Evaluation of suitability isn't required, can take sole option
            #to reduce startup time
            ffmpeg_path = ffmpeg_paths[0]
        else:
            ffmpeg_path = max(ffmpeg_paths, key=ffmpeg_suitability)

video_extensions = ['webm', 'mp4', 'mkv', 'gif', 'mov']

VHSLoadFormats = {
    'None': {},
    'AnimateDiff': {'target_rate': 8, 'dim': (8,0,512,512)},
    'Mochi': {'target_rate': 24, 'dim': (16,0,848,480), 'frames':(6,1)},
    'LTXV': {'target_rate': 24, 'dim': (32,0,768,512), 'frames':(8,1)},
    'Hunyuan': {'target_rate': 24, 'dim': (16,0,848,480), 'frames':(4,1)},
    'Cosmos': {'target_rate': 24, 'dim': (16,0,1280,704), 'frames':(8,1)},
    'Wan': {'target_rate': 16, 'dim': (8,0,832,480), 'frames':(4,1)},
}
"""
External plugins may add additional formats to nodes.VHSLoadFormats
In addition to shorthand options, direct widget names will map a given dict to options.
Adding a third arguement to a frames tuple can enable strict checks on number
of loaded frames, i.e (8,1,True)
"""
if not hasattr(nodes, 'VHSLoadFormats'):
    nodes.VHSLoadFormats = {}

def get_load_formats():
    #TODO: check if {**extra_config.VHSLoafFormats, **VHSLoadFormats} has minimum version
    formats = {}
    formats.update(nodes.VHSLoadFormats)
    formats.update(VHSLoadFormats)
    return (list(formats.keys()),
            {'default': 'AnimateDiff', 'formats': formats})
def get_format(format):
    if format in VHSLoadFormats:
        return VHSLoadFormats[format]
    return nodes.VHSLoadFormats.get(format, {})

def is_gif(filename) -> bool:
    file_parts = filename.split('.')
    return len(file_parts) > 1 and file_parts[-1] == "gif"


def target_size(width, height, custom_width, custom_height, downscale_ratio=8) -> tuple[int, int]:
    if downscale_ratio is None:
        downscale_ratio = 8
    if custom_width == 0 and custom_height ==  0:
        pass
    elif custom_height == 0:
        height *= custom_width/width
        width = custom_width
    elif custom_width == 0:
        width *= custom_height/height
        height = custom_height
    else:
        width = custom_width
        height = custom_height
    width = int(width/downscale_ratio + 0.5) * downscale_ratio
    height = int(height/downscale_ratio + 0.5) * downscale_ratio
    return (width, height)

def cv_frame_generator(video, force_rate, frame_load_cap, skip_first_frames,
                       select_every_nth, meta_batch=None, unique_id=None):
    video_cap = cv2.VideoCapture(video)
    if not video_cap.isOpened() or not video_cap.grab():
        raise ValueError(f"{video} could not be loaded with cv.")

    # extract video metadata
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    width = 0

    if width <=0 or height <=0:
        _, frame = video_cap.retrieve()
        height, width, _ = frame.shape

    # set video_cap to look at start_index frame
    total_frame_count = 0
    total_frames_evaluated = -1
    frames_added = 0
    base_frame_time = 1 / fps
    prev_frame = None

    if force_rate == 0:
        target_frame_time = base_frame_time
    else:
        target_frame_time = 1/force_rate

    if total_frames > 0:
        if force_rate != 0:
            yieldable_frames = int(total_frames / fps * force_rate)
        else:
            yieldable_frames = total_frames
        if select_every_nth:
            yieldable_frames //= select_every_nth
        if frame_load_cap != 0:
            yieldable_frames =  min(frame_load_cap, yieldable_frames)
    else:
        yieldable_frames = 0
    yield (width, height, fps, duration, total_frames, target_frame_time, yieldable_frames)
    pbar = ProgressBar(yieldable_frames)
    time_offset=target_frame_time
    while video_cap.isOpened():
        if time_offset < target_frame_time:
            is_returned = video_cap.grab()
            # if didn't return frame, video has ended
            if not is_returned:
                break
            time_offset += base_frame_time
        if time_offset < target_frame_time:
            continue
        time_offset -= target_frame_time
        # if not at start_index, skip doing anything with frame
        total_frame_count += 1
        if total_frame_count <= skip_first_frames:
            continue
        else:
            total_frames_evaluated += 1

        # if should not be selected, skip doing anything with frame
        if total_frames_evaluated%select_every_nth != 0:
            continue

        # opencv loads images in BGR format (yuck), so need to convert to RGB for ComfyUI use
        # follow up: can videos ever have an alpha channel?
        # To my testing: No. opencv has no support for alpha
        unused, frame = video_cap.retrieve()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # convert frame to comfyui's expected format
        # TODO: frame contains no exif information. Check if opencv2 has already applied
        frame = np.array(frame, dtype=np.float32)
        torch.from_numpy(frame).div_(255)
        if prev_frame is not None:
            inp  = yield prev_frame
            if inp is not None:
                #ensure the finally block is called
                return
        prev_frame = frame
        frames_added += 1
        if pbar is not None:
            pbar.update_absolute(frames_added, yieldable_frames)
        # if cap exists and we've reached it, stop processing frames
        if frame_load_cap > 0 and frames_added >= frame_load_cap:
            break
    if meta_batch is not None:
        meta_batch.inputs.pop(unique_id)
        meta_batch.has_closed_inputs = True
    if prev_frame is not None:
        yield prev_frame

#Python 3.12 adds an itertools.batched, but it's easily replicated for legacy support
def batched(it, n):
    while batch := tuple(itertools.islice(it, n)):
        yield batch
def batched_vae_encode(images, vae, frames_per_batch):
    for batch in batched(images, frames_per_batch):
        image_batch = torch.from_numpy(np.array(batch))
        yield from vae.encode(image_batch).numpy()
def resized_cv_frame_gen(custom_width, custom_height, downscale_ratio, **kwargs):
    gen = cv_frame_generator(**kwargs)
    info =  next(gen)
    width, height = info[0], info[1]
    frames_per_batch = (1920 * 1080 * 16) // (width * height) or 1
    if kwargs.get('meta_batch', None) is not None:
        frames_per_batch = min(frames_per_batch, kwargs['meta_batch'].frames_per_batch)
    if custom_width != 0 or custom_height != 0 or downscale_ratio is not None:
        new_size = target_size(width, height, custom_width, custom_height, downscale_ratio)
        yield (*info, new_size[0], new_size[1], False)
        if new_size[0] != width or new_size[1] != height:
            def rescale(frame):
                s = torch.from_numpy(np.fromiter(frame, np.dtype((np.float32, (height, width, 3)))))
                s = s.movedim(-1,1)
                s = common_upscale(s, new_size[0], new_size[1], "lanczos", "center")
                return s.movedim(1,-1).numpy()
            yield from itertools.chain.from_iterable(map(rescale, batched(gen, frames_per_batch)))
            return
    else:
        yield (*info, info[0], info[1], False)
    yield from gen

def load_video(meta_batch=None, unique_id=None, memory_limit_mb=None, vae=None,
               generator=resized_cv_frame_gen, format='None',  **kwargs):
    if 'force_size' in kwargs:
        kwargs.pop('force_size')
        logger.warn("force_size has been removed. Did you reload the webpage after updating?")
    format = get_format(format)
    kwargs['video'] = strip_path(kwargs['video'])
    if vae is not None:
        downscale_ratio = getattr(vae, "downscale_ratio", 8)
    else:
        downscale_ratio = format.get('dim', (1,))[0]
    if meta_batch is None or unique_id not in meta_batch.inputs:
        gen = generator(meta_batch=meta_batch, unique_id=unique_id, downscale_ratio=downscale_ratio, **kwargs)
        (width, height, fps, duration, total_frames, target_frame_time, yieldable_frames, new_width, new_height, alpha) = next(gen)

        if meta_batch is not None:
            meta_batch.inputs[unique_id] = (gen, width, height, fps, duration, total_frames, target_frame_time, yieldable_frames, new_width, new_height, alpha)
            if yieldable_frames:
                meta_batch.total_frames = min(meta_batch.total_frames, yieldable_frames)

    else:
        (gen, width, height, fps, duration, total_frames, target_frame_time, yieldable_frames, new_width, new_height, alpha) = meta_batch.inputs[unique_id]

    memory_limit = None
    if memory_limit_mb is not None:
        memory_limit *= 2 ** 20
    else:
        #TODO: verify if garbage collection should be performed here.
        #leaves ~128 MB unreserved for safety
        try:
            memory_limit = (psutil.virtual_memory().available + psutil.swap_memory().free) - 2 ** 27
        except:
            logger.warn("Failed to calculate available memory. Memory load limit has been disabled")
            memory_limit = BIGMAX
    if vae is not None:
        #space required to load as f32, exist as latent with wiggle room, decode to f32
        max_loadable_frames = int(memory_limit//(width*height*3*(4+4+1/10)))
    else:
        #TODO: use better estimate for when vae is not None
        #Consider completely ignoring for load_latent case?
        max_loadable_frames = int(memory_limit//(width*height*3*(.1)))
    if meta_batch is not None:
        if 'frames' in format:
            if meta_batch.frames_per_batch % format['frames'][0] != format['frames'][1]:
                error = (meta_batch.frames_per_batch - format['frames'][1]) % format['frames'][0]
                suggested = meta_batch.frames_per_batch - error
                if error > format['frames'][0] / 2:
                    suggested += format['frames'][0]
                raise RuntimeError(f"The chosen frames per batch is incompatible with the selected format. Try {suggested}")
        if meta_batch.frames_per_batch > max_loadable_frames:
            raise RuntimeError(f"Meta Batch set to {meta_batch.frames_per_batch} frames but only {max_loadable_frames} can fit in memory")
        gen = itertools.islice(gen, meta_batch.frames_per_batch)
    else:
        original_gen = gen
        gen = itertools.islice(gen, max_loadable_frames)
    frames_per_batch = (1920 * 1080 * 16) // (width * height) or 1
    if vae is not None:
        gen = batched_vae_encode(gen, vae, frames_per_batch)
        vw,vh = new_width//downscale_ratio, new_height//downscale_ratio
        channels = getattr(vae, 'latent_channels', 4)
        images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (channels,vh,vw)))))
    else:
        #Some minor wizardry to eliminate a copy and reduce max memory by a factor of ~2
        images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (new_height, new_width, 4 if alpha else 3)))))
    if meta_batch is None and memory_limit is not None:
        try:
            next(original_gen)
            raise RuntimeError(f"Memory limit hit after loading {len(images)} frames. Stopping execution.")
        except StopIteration:
            pass
    if len(images) == 0:
        raise RuntimeError("No frames generated")
    if 'frames' in format and len(images) % format['frames'][0] != format['frames'][1]:
        err_msg = f"The number of frames loaded {len(images)}, does not match the requirements of the currently selected format."
        if len(format['frames']) > 2 and format['frames'][2]:
            raise RuntimeError(err_msg)
        div, mod = format['frames'][:2]
        frames = (len(images) - mod) // div * div + mod
        images = images[:frames]
        #Commenting out log message since it's displayed in UI. consider further
        #logger.warn(err_msg + f" Output has been truncated to {len(images)} frames.")
    if 'start_time' in kwargs:
        start_time = kwargs['start_time']
    else:
        start_time = kwargs['skip_first_frames'] * target_frame_time
    target_frame_time *= kwargs.get('select_every_nth', 1)
    #Setup lambda for lazy audio capture
    audio = lazy_get_audio(kwargs['video'], start_time, kwargs['frame_load_cap']*target_frame_time)
    #Adjust target_frame_time for select_every_nth
    video_info = {
        "source_fps": fps,
        "source_frame_count": total_frames,
        "source_duration": duration,
        "source_width": width,
        "source_height": height,
        "loaded_fps": 1/target_frame_time,
        "loaded_frame_count": len(images),
        "loaded_duration": len(images) * target_frame_time,
        "loaded_width": new_width,
        "loaded_height": new_height,
    }
    if vae is None:
        return (images, len(images), audio, video_info)
    else:
        return ({"samples": images}, len(images), audio, video_info)