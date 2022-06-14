import os.path
from utils import files_utils, train_utils, image_utils
import dlib
import tqdm
import face_alignment
from custom_types import *
import imageio
import PIL
from PIL import Image
import scipy
from scipy.ndimage.filters import gaussian_filter1d
import constants
import ffmpeg
import cv2
from argparse import ArgumentParser

def get_drawer(img, shape, line_width):


    def draw_curve_(idx_list, color=(0, 255, 0), loop=False):
        for i in idx_list:
            cv2.line(img, (shape[i, 0], shape[i, 1]), (shape[i + 1, 0], shape[i + 1, 1]), color, line_width)
        if (loop):
            cv2.line(img, (shape[idx_list[0], 0], shape[idx_list[0], 1]),
                     (shape[idx_list[-1] + 1, 0], shape[idx_list[-1] + 1, 1]), color, line_width)

    return draw_curve_


def vis_landmark_on_img(img, shape, line_width=2):
    '''
    Visualize landmark on images.
    '''


    shape = shape.astype(np.int32)
    draw_curve = get_drawer(img, shape, line_width)
    draw_curve(list(range(0, 16)), color=(255, 144, 25))  # jaw
    draw_curve(list(range(17, 21)), color=(50, 205, 50))  # eye brow
    draw_curve(list(range(22, 26)), color=(50, 205, 50))
    draw_curve(list(range(27, 35)), color=(208, 224, 63))  # nose
    draw_curve(list(range(36, 41)), loop=True, color=(71, 99, 255))  # eyes
    draw_curve(list(range(42, 47)), loop=True, color=(71, 99, 255))
    draw_curve(list(range(48, 59)), loop=True, color=(238, 130, 238))  # mouth
    draw_curve(list(range(60, 67)), loop=True, color=(238, 130, 238))
    return img


def crop_image(filepath, quad, enable_padding=False, image_size=1024):
    x = (quad[3] - quad[1]) / 2
    qsize = np.hypot(*x) * 2
    # read image
    if isinstance(filepath, PIL.Image.Image):
        img = filepath
    elif isinstance(filepath, ARRAY):
        img = PIL.Image.fromarray(filepath)
    else:
        img = PIL.Image.open(filepath)
    transform_size = image_size
    # Shrink.
    shrink = int(np.floor(qsize / image_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink
    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if (crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]):
        img = img.crop(crop)
        quad -= crop[0:2]
    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]
    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if image_size < transform_size:
        img = img.resize((image_size, image_size), PIL.Image.ANTIALIAS)
    return img


def calc_alignment_coefficients(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    a = np.matrix(matrix, dtype=float)
    b = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(a.T * a) * a.T, b)
    return np.array(res).reshape(8)


class FaceAlign:

    def get_landmarks_cached(self, path: List[str], image: Optional[ARRAY] = None) -> ARRAY:
        name = path[0].split('/')[-2:] + [path[1]]
        name = 'landmarks_'.join(name)
        # path_cache = f"{constants.MNT_ROOT}/cache/{name}.npy"
        path_cache = "/mnt/ml/projects/dubbing/cache/template_landmarks.npy"
        if files_utils.is_file(path_cache):
            landmarks = files_utils.load_np(path_cache)
        else:
            if image is None:
                if path[-1] == '.npy':
                    image = files_utils.load_np(''.join(path))
                else:
                    image = files_utils.load_image(''.join(path))
            landmarks = self.get_landmarks(image)
            files_utils.save_np(landmarks, path_cache)
        return landmarks


    def get_landmarks(self, image):
        """get landmark with dlib
        :return: np.array shape=(68, 2)
        """
        if self.fa is not None:
            lms, _, bboxes = self.fa.get_landmarks(image, return_bboxes=True)
            if len(lms) == 0:
                return None
            return lms[0]

        if self.detector is None:
            self.detector = dlib.get_frontal_face_detector()
        dets = self.detector(image)

        for k, d in enumerate(dets):
            shape = self.predictor(image, d)
            break
        else:
            return None
        t = list(shape.parts())
        a = []
        for tt in t:
            a.append([tt.x, tt.y])
        lm = np.array(a)
        return lm

    def compute_transform(self, image, scale=1.0):
        lm = self.get_landmarks(image)
        if lm is None:
            raise ValueError(f'Did not detect any faces in image')

        lm_eye_left = lm[36: 42]  # left-clockwise
        lm_eye_right = lm[42: 48]  # left-clockwise
        lm_mouth_outer = lm[48: 60]  # left-clockwise
        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg
        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)

        x *= scale
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        return lm, c, x, y

    def crop_faces_by_quads(self, vid, quads):
        orig_images, crops = [], []
        self.logger.start(len(quads))
        for i, quad in enumerate(quads):
            frame = vid.get_data(i)
            crop = crop_image(frame, quad.copy())
            orig_images.append(frame)
            crops.append(crop)
            self.logger.reset_iter()
        self.logger.stop()
        return crops, orig_images

    def crop_face(self, frame, image_size=1024):
        lm, c, x, y = self.compute_transform(frame, scale=1.)
        quad = V([c - x - y, c - x + y, c + x + y, c + x - y])
        crop = V(crop_image(frame, quad.copy(), image_size=image_size))
        lm_crop = self.compute_transform(crop, scale=1.)[0]
        return crop, quad, lm, lm_crop

    def crop_faces(self, vid, num_frames, scale=1., center_sigma=1., xy_sigma=3.0, use_fa=False):
        cs, xs, ys = [], [], []
        self.logger.start(num_frames)
        for i in range(num_frames):
            frame = vid.get_data(i)
            try:
                lm, c, x, y = self.compute_transform(frame, scale=scale)
                self.errors = 0
                self.cache = lm, c, x, y
            except ValueError:
                self.errors += 1
                if self.errors > 3:
                    raise ValueError
                lm, c, x, y = self.cache
            cs.append(c)
            xs.append(x)
            ys.append(y)
            self.logger.reset_iter()
        self.logger.stop()
        cs = np.stack(cs)
        xs = np.stack(xs)
        ys = np.stack(ys)
        if center_sigma != 0:
            cs = gaussian_filter1d(cs, sigma=center_sigma, axis=0)

        if xy_sigma != 0:
            xs = gaussian_filter1d(xs, sigma=xy_sigma, axis=0)
            ys = gaussian_filter1d(ys, sigma=xy_sigma, axis=0)

        quads = np.stack([cs - xs - ys, cs - xs + ys, cs + xs + ys, cs + xs - ys], axis=1)
        crops, orig_images = self.crop_faces_by_quads(vid, list(quads))

        return crops, orig_images, quads

    @staticmethod
    def output_sound(video_path, out_path):
        audio_path = f'{out_path}/audio.wav'
        if not files_utils.is_file(audio_path):
            files_utils.init_folders(audio_path)
            in1 = ffmpeg.input(video_path)
            a1 = in1.audio
            out = ffmpeg.output(a1, audio_path)
            out.run()

    def crop_video(self, video_path: str, out_path: str, max_len: int):
        vid = imageio.get_reader(f"{video_path}", 'ffmpeg')
        fps = vid._meta['fps']
        self.output_sound(video_path, out_path)
        num_frames = min(int(fps * max_len), vid.count_frames())
        metadata = {'fps': fps, 'frames_count': vid.count_frames(), 'actual_frames': num_frames}
        crops, orig_images, quads = self.crop_faces(vid, num_frames)
        files_utils.save_np(quads, f'{out_path}/quads')
        image_utils.gif_group(crops, f'{out_path}/orig_align', fps)
        # landmarks = np.zeros((len(crops), 68, 2))
        for i, crop in enumerate(crops):
            # landmarks[i] = self.get_3d_landmakrs(crop)
            files_utils.save_np(crop, f'{out_path}/crop_{i:04d}')
        metadata["video_path"] = video_path
        files_utils.save_pickle(metadata, f'{out_path}/metadata')

    def landmarks_only(self, out_path, dim=3):
        metadata = files_utils.load_pickle(f'{out_path}/metadata')
        key = 'landmarks' if dim==3 else 'landmarks_2d'
        if key in metadata:
            return
        paths = files_utils.collect(out_path, '.npy')
        paths = [path for path in paths if 'crop' in path[1]]
        landmarks = np.zeros((len(paths), 68, dim))
        self.logger.start(len(paths))
        for i, path in enumerate(paths):
            crop = files_utils.load_np(''.join(path))
            # landmark_2d = self.get_landmarks(crop)
            if dim == 3:
                landmarks[i] = self.predictor_3d.get_landmarks(crop)[0]
            else:
                landmarks[i] = self.get_landmarks(crop)
            self.logger.reset_iter()
            # image_a = vis_landmark_on_img(crop.copy(), landmark_2d)
            # image_b = vis_landmark_on_img(crop.copy(), landmark_3d)
            # files_utils.imshow(image_a)
            # files_utils.imshow(image_b)
        metadata[key] = landmarks
        files_utils.save_pickle(metadata, f'{out_path}/metadata')
        self.logger.stop()



    @staticmethod
    def paste_image(quad, image_new, image_orig):
        image_size = image_new.shape[0]
        inverse_transform = calc_alignment_coefficients(quad + 0.5, [[0, 0], [0, image_size], [image_size, image_size],
                                                                     [image_size, 0]])
        image_new = Image.fromarray(image_new).convert('RGBA')
        image_orig = Image.fromarray(image_orig).convert('RGBA')
        paste = image_new.transform(image_orig.size, Image.PERSPECTIVE, inverse_transform, Image.BILINEAR)
        image_orig.paste(paste, (0, 0), mask=paste)
        image_orig = V(image_orig)[:, :, :3]
        return image_orig

    def uncrop_video(self, crops_root, video_new: Union[str, ARRAYS], starting_time=-1) -> ARRAYS:

        class DummyVid:

            def count_frames(self):
                return len(self.lst)

            def get_data(self, i):
                return self.lst[i]

            def __init__(self, lst):
                self.lst = lst
        metadata = files_utils.load_pickle(f"{crops_root}/metadata")
        video_path = metadata["video_path"].replace("/home/ahertz/projects/StyleFusion-main/assets/",
                                                    "/mnt/ml/projects/")
        vid_orig = imageio.get_reader(f"{video_path}", 'ffmpeg')
        if starting_time > 0:
            prefix = int(metadata['fps'] * starting_time)
        else:
            prefix = 0
        if type(video_new) is str:
            vid_new = imageio.get_reader(f"{video_new}", 'ffmpeg')
        else:
            vid_new = DummyVid(video_new)
        quads = files_utils.load_np(f'{crops_root}/quads')

        num_frames = min(vid_orig.count_frames(), vid_new.count_frames())
        seq_paste = []
        for i in range(num_frames):
            quad = quads[prefix + i]
            frame_orig, frame_new = vid_orig.get_data(prefix + i), vid_new.get_data(i)
            pasted = self.paste_image(quad, frame_new, frame_orig)
            seq_paste.append(pasted)
        return seq_paste

    @property
    def predictor_3d(self) -> face_alignment.FaceAlignment:
        if self.predictor_3d_ is None:
            self.predictor_3d_ = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda', flip_input=True)
        return self.predictor_3d_

    def __init__(self):
        self.predictor = dlib.shape_predictor(f"{constants.PROJECT_ROOT}/weights/ffhq/shape_predictor_68_face_landmarks.dat")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor_3d_: Optional[face_alignment.FaceAlignment] = None
        self.fa = None
        self.logger = train_utils.Logger()
        self.errors = 0
        self.cache = None
        # self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)


if __name__ == '__main__':
    aligner = FaceAlign()
    # aligner.uncrop_video(f"/mnt/r/projects/facesimile/shots/101/purpledino_genwoman/comp_wip/images/comp_main_publish_qt/101_purpledino_genwoman_comp_v010.mov",
    #                      f"{constants.DATA_ROOT}/101_purpledino_genwoman_comp_v010", f"{constants.DATA_ROOT}/101_purpledino_genwoman_comp_v010/orig_align.mp4")
    # aligner.crop_video(f"/mnt/r/projects/facesimile/shots/101/purpledino_genwoman/comp_wip/images/comp_main_publish_qt/101_purpledino_genwoman_comp_v017.mov",
    #                    f"{constants.DATA_ROOT}/101_purpledino_genwoman_comp_v017", 100)

    # aligner.crop_video(f"{constants.MNT_ROOT}/marz_dub/BillieFrench_FaceFormer.mp4",
    #                    f"{constants.MNT_ROOT}/processed_infer/BillieFrench_FaceFormer", 10000)
    # aligner.landmarks_only(f"{constants.MNT_ROOT}/processed_infer/BillieFrench_FaceFormer", 2)
    # aligner.landmarks_only(f"{constants.MNT_ROOT}/processed_infer/BillieFrench_FaceFormer", 3)
    input_dir = "/mnt/ml/projects/dubbing/marz_dub/project2/FaceFormer Outputs"
    output_dir = "/mnt/ml/projects/dubbing/processed_infer"
    files = os.listdir(input_dir)
    for file in files:
        input_file = os.path.join(input_dir, file)
        output_file = os.path.join(output_dir, file[:-4] + "_FaceFormer")
        print(input_file, output_file)
        aligner.crop_video(input_file, output_file, 10000)
        aligner.landmarks_only(output_file, 2)
        aligner.landmarks_only(output_file, 3)
    # aligner.crop_video(f"{constants.DATA_ROOT}/raw_videos/obama_062814.mp4", f"{constants.DATA_ROOT}/obama", 30)
    # viseme2vec(f"{constants.DATA_ROOT}/raw_videos/obama_062814",
    #             f"{constants.DATA_ROOT}/processed/viseme_obama_062814",
    #            f"{constants.DATA_ROOT}/processed/viseme_vec_obama_062814")
    # p2fa_folder(f"{constants.DATA_ROOT}/processed/")
    # split_audio(f"{constants.DATA_ROOT}/raw_videos/obama_062814",
    #             f"{constants.DATA_ROOT}/processed/obama_062814")
    # split_text(f"{constants.DATA_ROOT}/raw_videos/obama_062814.txt",
    #            f"{constants.DATA_ROOT}/processed/obama_062814.txt")
