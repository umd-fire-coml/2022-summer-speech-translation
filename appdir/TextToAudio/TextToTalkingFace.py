import os, glob
from base64 import b64encode
from gtts import gTTS
import gdown
import threading
import shutil
import zipfile

##PARSER STUFF
ADD_NAIVE_EYE = True  # whether add naive eye blink
CLOSE_INPUT_FACE_MOUTH = False  # if your image has an opened mouth, put this as True, else False
AMP_LIP_SHAPE_X = 2.  # amplify the lip motion in horizontal direction
AMP_LIP_SHAPE_Y = 2.  # amplify the lip motion in vertical direction
AMP_HEAD_POSE_MOTION = 0.5  # amplify the head pose motion (usually smaller than 1.0, put it to 0. for a static head pose)


### IMPORTANT: THIS STUFF ONLY WORKS IF THE SCRIPT IS RUN IN APP.PY BECAUSE OF PATHS
def addMainFolder():
    inp = 'https://drive.google.com/file/d/1R-cvca7VuHCE5dbzUiFPjzBroqSVk49z/view?usp=sharing'

    # Set Output
    out = 'zip.zip'

    # Download
    if not os.path.exists(out):
        gdown.download(url = inp, output = out, quiet = False, fuzzy=True)
    if not os.path.exists("doc"):
        with zipfile.ZipFile(out, 'r') as zip_ref:
            zip_ref.extractall()
            print("extract done")


addMainFolder()
import sys
sys.path.append("thirdparty/AdaptiveWingLoss")
import numpy as np
import cv2
import argparse
from src.approaches.train_image_translation import Image_translation_block
import torch
import pickle
import face_alignment
from src.autovc.AutoVC_mel_Convertor_retrain_version import AutoVC_mel_Convertor
import shutil
import time
import util.utils as util
from scipy.signal import savgol_filter
from src.approaches.train_audio2landmark import Audio2landmark_model
from IPython.display import HTML


def downloadCKPTS():
    # PATH TO CKPT
    ckpt_autovc = 'https://drive.google.com/uc?id=1ZiwPp_h62LtjU0DwpelLUoodKPR85K7x'
    ckpt_content_branch = 'https://drive.google.com/uc?id=1r3bfEvTVl6pCNw5xwUhEglwDHjWtAqQp'
    ckpt_speaker_branch = 'https://drive.google.com/uc?id=1rV0jkyDqPW-aDJcj7xSO6Zt1zSXqn1mu'
    ckpt_116_i2i_comb = 'https://drive.google.com/uc?id=1i2LJXKp-yWKIEEgJ7C6cE3_2NirfY_0a'
    emb_pickle = 'https://drive.google.com/uc?id=18-0CYl5E6ungS3H4rRSHjfYvvm-WwjTI'

    # Set Output
    ckpt_autovc_out = 'examples/ckpt/ckpt_autovc.pth'
    ckpt_content_branch_out = 'examples/ckpt/ckpt_content_branch.pth'
    ckpt_speaker_branch_out = 'examples/ckpt/ckpt_speaker_branch.pth'
    ckpt_116_i2i_comb_out = 'examples/ckpt/ckpt_116_i2i_comb.pth'
    emb_pickle_out = 'examples/dumpemb.pickle'

    # Download
    gdown.download(url = ckpt_autovc, output = ckpt_autovc_out, quiet = False, fuzzy = True)
    gdown.download(url = ckpt_content_branch, output = ckpt_content_branch_out, quiet = False, fuzzy = True)
    gdown.download(url = ckpt_speaker_branch, output = ckpt_speaker_branch_out, quiet = False, fuzzy = True)
    gdown.download(url = ckpt_116_i2i_comb, output = ckpt_116_i2i_comb_out, quiet = False, fuzzy = True)
    gdown.download(url = emb_pickle, output = emb_pickle_out, quiet = False, fuzzy = True)

def TextToAudio(txt_str, save_pth):
  tts = gTTS(txt_str)
  audio_file = save_pth
  tts.save(audio_file)
  return audio_file

list_of_faces = ['angelina', 'anne', 'audrey', 'aya', 'cesi', 'dali',
                 'donald', 'dragonmom', 'dwayne', 'harry', 'hermione',
                 'johnny', 'leo', 'morgan', 'natalie', 'neo', 'obama',
                 'rihanna', 'ron', 'scarlett', 'taylor']

def getListofFaceOptions():
  return list_of_faces

def getParser(default_head_name):
  parser = argparse.ArgumentParser(default_head_name)
  parser.add_argument('--jpg', type=str, default='{}.jpg'.format(default_head_name))
  parser.add_argument('--close_input_face_mouth', default=CLOSE_INPUT_FACE_MOUTH, action='store_true')

  parser.add_argument('--load_AUTOVC_name', type=str, default='examples/ckpt/ckpt_autovc.pth')
  parser.add_argument('--load_a2l_G_name', type=str, default='examples/ckpt/ckpt_speaker_branch.pth')
  parser.add_argument('--load_a2l_C_name', type=str, default='examples/ckpt/ckpt_content_branch.pth')
  parser.add_argument('--load_G_name', type=str, default='examples/ckpt/ckpt_116_i2i_comb.pth')

  parser.add_argument('--amp_lip_x', type=float, default=AMP_LIP_SHAPE_X)
  parser.add_argument('--amp_lip_y', type=float, default=AMP_LIP_SHAPE_Y)
  parser.add_argument('--amp_pos', type=float, default=AMP_HEAD_POSE_MOTION)
  parser.add_argument('--reuse_train_emb_list', type=str, nargs='+', default=[])
  parser.add_argument('--add_audio_in', default=False, action='store_true')
  parser.add_argument('--comb_fan_awing', default=False, action='store_true')
  parser.add_argument('--output_folder', type=str, default='examples')

  parser.add_argument('--test_end2end', default=True, action='store_true')
  parser.add_argument('--dump_dir', type=str, default='', help='')
  parser.add_argument('--pos_dim', default=7, type=int)
  parser.add_argument('--use_prior_net', default=True, action='store_true')
  parser.add_argument('--transformer_d_model', default=32, type=int)
  parser.add_argument('--transformer_N', default=2, type=int)
  parser.add_argument('--transformer_heads', default=2, type=int)
  parser.add_argument('--spk_emb_enc_size', default=16, type=int)
  parser.add_argument('--init_content_encoder', type=str, default='')
  parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
  parser.add_argument('--reg_lr', type=float, default=1e-6, help='weight decay')
  parser.add_argument('--write', default=False, action='store_true')
  parser.add_argument('--segment_batch_size', type=int, default=1, help='batch size')
  parser.add_argument('--emb_coef', default=3.0, type=float)
  parser.add_argument('--lambda_laplacian_smooth_loss', default=1.0, type=float)
  parser.add_argument('--use_11spk_only', default=False, action='store_true')
  parser.add_argument('-f')

  opt_parser = parser.parse_args()

  return opt_parser

def TextToTalkingFace(txt_string, face_name):
  # Get Audio fo txt_string
  audio_path = 'examples/speech.wav'
  TextToAudio(txt_string, audio_path)

  # Face Animation
  default_head_name = face_name        # the image name (with no .jpg) to animate

  opt_parser = getParser(default_head_name)

  img =cv2.imread('examples/' + opt_parser.jpg)
  predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=True)
  shapes = predictor.get_landmarks(img)
  if (not shapes or len(shapes) != 1):
      exit(-1)
  shape_3d = shapes[0]

  if(opt_parser.close_input_face_mouth):
      util.close_input_face_mouth(shape_3d)

  shape_3d[48:, 0] = (shape_3d[48:, 0] - np.mean(shape_3d[48:, 0])) * 1.05 + np.mean(shape_3d[48:, 0]) # wider lips
  shape_3d[49:54, 1] += 0.           # thinner upper lip
  shape_3d[55:60, 1] -= 1.           # thinner lower lip
  shape_3d[[37,38,43,44], 1] -=2.    # larger eyes
  shape_3d[[40,41,46,47], 1] +=2.    # larger eyes

  shape_3d, scale, shift = util.norm_input_face(shape_3d)

  au_data = []
  au_emb = []
  ains = glob.glob1('examples', '*.wav')
  ains = [item for item in ains if item != 'tmp.wav']
  ains.sort()
  for ain in ains:
      os.system('ffmpeg -y -loglevel error -i examples/{} -ar 16000 examples/tmp.wav'.format(ain))
      shutil.copyfile('examples/tmp.wav', 'examples/{}'.format(ain))

      # au embedding
      from thirdparty.resemblyer_util.speaker_emb import get_spk_emb
      me, ae = get_spk_emb('examples/{}'.format(ain))
      au_emb.append(me.reshape(-1))

      c = AutoVC_mel_Convertor('examples')

      au_data_i = c.convert_single_wav_to_autovc_input(audio_filename=os.path.join('examples', ain),
            autovc_model_path=opt_parser.load_AUTOVC_name)
      au_data += au_data_i
  if(os.path.isfile('examples/tmp.wav')):
      os.remove('examples/tmp.wav')

  # landmark fake placeholder
  fl_data = []
  rot_tran, rot_quat, anchor_t_shape = [], [], []
  for au, info in au_data:
      au_length = au.shape[0]
      fl = np.zeros(shape=(au_length, 68 * 3))
      fl_data.append((fl, info))
      rot_tran.append(np.zeros(shape=(au_length, 3, 4)))
      rot_quat.append(np.zeros(shape=(au_length, 4)))
      anchor_t_shape.append(np.zeros(shape=(au_length, 68 * 3)))

  if(os.path.exists(os.path.join('examples', 'dump', 'random_val_fl.pickle'))):
      os.remove(os.path.join('examples', 'dump', 'random_val_fl.pickle'))
  if(os.path.exists(os.path.join('examples', 'dump', 'random_val_fl_interp.pickle'))):
      os.remove(os.path.join('examples', 'dump', 'random_val_fl_interp.pickle'))
  if(os.path.exists(os.path.join('examples', 'dump', 'random_val_au.pickle'))):
      os.remove(os.path.join('examples', 'dump', 'random_val_au.pickle'))
  if (os.path.exists(os.path.join('examples', 'dump', 'random_val_gaze.pickle'))):
      os.remove(os.path.join('examples', 'dump', 'random_val_gaze.pickle'))

  with open(os.path.join('examples', 'dump', 'random_val_fl.pickle'), 'wb') as fp:
      pickle.dump(fl_data, fp)
  with open(os.path.join('examples', 'dump', 'random_val_au.pickle'), 'wb') as fp:
      pickle.dump(au_data, fp)
  with open(os.path.join('examples', 'dump', 'random_val_gaze.pickle'), 'wb') as fp:
      gaze = {'rot_trans':rot_tran, 'rot_quat':rot_quat, 'anchor_t_shape':anchor_t_shape}
      pickle.dump(gaze, fp)

  model = Audio2landmark_model(opt_parser, jpg_shape=shape_3d)
  if(len(opt_parser.reuse_train_emb_list) == 0):
      model.test(au_emb=au_emb)
  else:
      model.test(au_emb=None)

  fls = glob.glob1('examples', 'pred_fls_*.txt')
  fls.sort()

  for i in range(0,len(fls)):
      fl = np.loadtxt(os.path.join('examples', fls[i])).reshape((-1, 68,3))
      fl[:, :, 0:2] = -fl[:, :, 0:2]
      fl[:, :, 0:2] = fl[:, :, 0:2] / scale - shift

      if (ADD_NAIVE_EYE):
          fl = util.add_naive_eye(fl)

      # additional smooth
      fl = fl.reshape((-1, 204))
      fl[:, :48 * 3] = savgol_filter(fl[:, :48 * 3], 15, 3, axis=0)
      fl[:, 48*3:] = savgol_filter(fl[:, 48*3:], 5, 3, axis=0)
      fl = fl.reshape((-1, 68, 3))

      ''' STEP 6: Imag2image translation '''
      model = Image_translation_block(opt_parser, single_test=True)
      with torch.no_grad():
          model.single_test(jpg=img, fls=fl, filename=fls[i], prefix=opt_parser.jpg.split('.')[0])
      os.remove(os.path.join('examples', fls[i]))

  for ain in ains:
    OUTPUT_MP4_NAME = '{}_pred_fls_{}_audio_embed.mp4'.format(
      opt_parser.jpg.split('.')[0],
      ain.split('.')[0]
      )

    mp4 = open('examples/{}'.format(OUTPUT_MP4_NAME),'rb').read()

    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

    return 'examples/' + str(OUTPUT_MP4_NAME)

def deleteOldFiles(audio_name):
  os.remove('examples/speech.wav')
  os.remove('examples/speech_av.mp4')
  os.remove(audio_name)


