# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

#----------------------------------------------------------------------------
# Convenience class that behaves exactly like dict(), but allows accessing
# the keys and values using the attribute syntax, i.e., "mydict.key = value".

class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]

#----------------------------------------------------------------------------
# Paths.

data_dir = '.'
result_dir = 'results'

#----------------------------------------------------------------------------
# TensorFlow options.

tf_config = EasyDict()  # TensorFlow session config, set by tfutil.init_tf().
env = EasyDict()        # Environment variables, set by the main program in train.py.

tf_config['graph_options.place_pruned_graph']   = True      # False (default) = Check that all ops are available on the designated device. True = Skip the check for ops that are not used.
tf_config['gpu_options.allow_growth']           = True     # False (default) = Allocate all GPU memory at the beginning. True = Allocate only as much GPU memory as needed.
#env.CUDA_VISIBLE_DEVICES                       = '0,1,2,3'       # Unspecified (default) = Use all available GPUs. List of ints = CUDA device numbers to use.
env.TF_CPP_MIN_LOG_LEVEL                        = '1'       # 0 (default) = Print all available debug info from TensorFlow. 1 = Print warnings and errors, but disable debug info.

#----------------------------------------------------------------------------
# Official training configs, targeted mainly for CelebA-HQ.
# To run, comment/uncomment the lines as appropriate and launch train.py.

desc        = 'pgan'                                           # Description string included in result subdir name.
random_seed = 1000                                             # Global random seed.
dataset     = EasyDict()                                       # Options for dataset.load_dataset().


train       = EasyDict(func='train_3D.train_progressive_gan')  # Options for main training func.

#train       = EasyDict(func='train_3D.train_progressive_gan', resume_run_id=34, resume_snapshot=4854, resume_kimg=4854.8)

#train       = EasyDict(func='train_3D.train_progressive_gan', resume_run_id=15, resume_snapshot=5974, resume_kimg=5974.0)

#train       = EasyDict(func='train_3D.train_progressive_gan', resume_run_id=16, resume_snapshot=6874, resume_kimg=6874.0)

# This is what I used before
#train       = EasyDict(func='train_3D.train_progressive_gan', resume_run_id=55, resume_snapshot=7354, resume_kimg=7354.0)

#train       = EasyDict(func='train_3D.train_progressive_gan', resume_run_id=37, resume_snapshot=7214, resume_kimg=7214.0)


## New runs

#train       = EasyDict(func='train_3D.train_progressive_gan', resume_run_id=66, resume_snapshot=4946, resume_kimg=4946.1)

#train       = EasyDict(func='train_3D.train_progressive_gan', resume_run_id=70, resume_snapshot=6992, resume_kimg=6992.1)

##

G           = EasyDict(func='networks_3D.G_paper')             # Options for generator network.
D           = EasyDict(func='networks_3D.D_paper')             # Options for discriminator network.
G_opt       = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)    # Options for generator optimizer.
D_opt       = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)    # Options for discriminator optimizer.
G_loss      = EasyDict(func='loss_3D.G_wgan_acgan')            # Options for generator loss.
D_loss      = EasyDict(func='loss_3D.D_wgangp_acgan')          # Options for discriminator loss.
D_accuracy  = EasyDict(func='loss_3D.D_wgangp_acgan_accuracy') # Options for discriminator accuracy
sched       = EasyDict()                                       # Options for train.TrainingSchedule.
grid        = EasyDict(size='1080p', layout='random')          # Options for train.setup_snapshot_image_grid().

# Dataset (choose one).

#desc += '-HCP_T1T2_64cubes';            dataset = EasyDict(tfrecord_dir='HCP_T1T2_64cubes'); train.mirror_augment = False

desc += '-MOUSE';            dataset = EasyDict(tfrecord_dir='mice-1-29-half-res'); train.mirror_augment = True

# Conditioning & snapshot options.
#desc += '-cond'; dataset.max_label_size = 'full' # conditioned on full label
#desc += '-cond1'; dataset.max_label_size = 1 # conditioned on first component of the label
#desc += '-g4k'; grid.size = '4k'
#desc += '-grpc'; grid.layout = 'row_per_class'

# Config presets (choose one).

desc += '-preset-v2-1gpu'; num_gpus = 1; sched.minibatch_base = 4; sched.G_lrate_dict = {1024: 0.0015}; sched.D_lrate_dict = EasyDict(sched.G_lrate_dict); train.total_kimg = 120000
sched.minibatch_dict = {4: 64, 8: 32, 16: 16, 32: 16, 64: 4}
sched.tick_kimg_dict = {4: 200, 8:200, 16:200, 32:100, 64:100}

# 128 fmap_max, 8192 fmap_base for 000-pgan-MOUSE-preset-v2-1gpu-fp32
# ====== MODIFY THESE ======
sched.lod_training_kimg = 500; sched.lod_transition_kimg = 500
G.fmap_max = 256; G.fmap_base = 2048; D.fmap_max = 256; D.fmap_base = 2048; G.latent_size = 1024

#desc += '-preset-v2-2gpus'; num_gpus = 2; sched.minibatch_base = 8; sched.minibatch_dict = {4: 4096, 8: 2048, 16: 256, 32: 64, 64: 16, 128: 2, 256: 2}; sched.G_lrate_dict = {512: 0.0015, 1024: 0.002}; sched.D_lrate_dict = EasyDict(sched.G_lrate_dict); train.total_kimg = 120000


#---
# Works for 128 filters but not so good images, lacks detail, low diversity, used 1000 volumes per level, results in 004 ...

#desc += '-preset-v2-4gpus'; num_gpus = 4; sched.minibatch_base = 4; sched.minibatch_dict = {4: 512, 8: 256, 16: 16, 32: 4, 64: 4, 128: 4}; sched.G_lrate_dict = {4: 0.0003, 8: 0.0003, 16: 0.0006, 32: 0.001, 64: 0.002, 256: 0.0015, 512: 0.002, 1024: 0.003}; sched.D_lrate_dict = EasyDict(sched.G_lrate_dict); train.total_kimg = 15000
#---
# Works well for 128 random subvolumes up to kimg 10760, used 2000 volumes per level, results in 010

#desc += '-preset-v2-4gpus'; num_gpus = 4; sched.minibatch_base = 4; sched.minibatch_dict = {4: 512, 8: 256, 16: 16, 32: 4, 64: 4, 128: 4}; sched.G_lrate_dict = {4: 0.0003, 8: 0.0003, 16: 0.0006, 32: 0.0015, 64: 0.0025, 128: 0.0035};  sched.D_lrate_dict = EasyDict(sched.G_lrate_dict); train.total_kimg = 40000     #sched.D_lrate_dict = {4: 0.0003, 8: 0.0003, 16: 0.0006, 32: 0.001, 64: 0.002, 128: 0.003}  ; train.total_kimg = 15000
#---

# Works well up to 10360, then volumes become blurry and very similar to each other

#desc += '-preset-v2-4gpus'; num_gpus = 4; sched.minibatch_base = 4; sched.minibatch_dict = {4: 512, 8: 256, 16: 16, 32: 4, 64: 4, 128: 4}; sched.G_lrate_dict = {4: 0.0003, 8: 0.0003, 16: 0.0006, 32: 0.001, 64: 0.0015, 128: 0.0025};  sched.D_lrate_dict = EasyDict(sched.G_lrate_dict); train.total_kimg = 40000     #sched.D_lrate_dict = {4: 0.0003, 8: 0.0003, 16: 0.0006, 32: 0.001, 64: 0.002, 128: 0.003}  ; train.total_kimg = 15000
#---

# Started from 10260, works well up to 11360, then fails

#desc += '-preset-v2-4gpus'; num_gpus = 4; sched.minibatch_base = 4; sched.minibatch_dict = {4: 512, 8: 256, 16: 16, 32: 4, 64: 4, 128: 4}; sched.G_lrate_dict = {4: 0.0003, 8: 0.0003, 16: 0.0006, 32: 0.001, 64: 0.0015, 128: 0.0025};  sched.D_lrate_dict = EasyDict(sched.G_lrate_dict); sched.D_lrate_dict = {4: 0.0003, 8: 0.0003, 16: 0.0006, 32: 0.0008, 64: 0.0012, 128: 0.002}  ; train.total_kimg = 40000
#------

# Started from 11460, works well to 12560

#desc += '-preset-v2-4gpus'; num_gpus = 4; sched.minibatch_base = 4; sched.minibatch_dict = {4: 512, 8: 256, 16: 16, 32: 4, 64: 4, 128: 4}; sched.G_lrate_dict = {4: 0.0003, 8: 0.0003, 16: 0.0006, 32: 0.0006, 64: 0.0012, 128: 0.0025}; sched.D_lrate_dict = EasyDict(sched.G_lrate_dict); sched.D_lrate_dict = {4: 0.0003, 8: 0.0003, 16: 0.0006, 32: 0.0006, 64: 0.0010, 128: 0.002}  ; train.total_kimg = 40000
#-------

# Started from scratch again, now with 1000 images per level, otherwise the learning rate needs to be very low, results in 015

# still too high learning rate for 32, lowering to 0.00006 to get through to next scale, results in 16

# learning rate 0.0005 for 64 seems too high, lowering to 0.0003, results in 17

# 0.0003 is still too high, lowering to 0.00015, results in 18

# results are not good for 64, increasing learning rate to 0.0003 for generator but keeping it at 0.00015 for the discriminator, results in 19

# still bad results, increasing LR to 0.0006 for generator and 0.0003 for discriminator, results in 20, a bit weird results, a bit too non-linear brains

# increasing LR to 0.0009 for generator and 0.0007 for discriminator, results in 21

# testing 0.0001 for G and D, results in 22

# results are starting to look better but still some artefacts, changing batch size to 8 as a way to reduce the number of updates / learning rate, results in 23

# increasing batch size to 16, results in 24, still creates strange artefacts at edge of brain



# starting training from 6874 instead, since optimization is reset for a new LOD , increasing batch size for 32 from 4 to 32, results in 25

# lowering learning rate to 0.00003, results in 26

# lowering batch size back to 4 for resolution 32, results in 27

# increasing learning rate for 64 to 0.0001, results in 28

# increasing learning rate for 64 to 0.0003, results in 29

# increasing learning rate for 64 to 0.0006, results in 30, works well until 7294, images then become very smooth

# increasing learning rate for 32 to 0.0006, since all layers remain trainable, and batch for 32 from 4 to 8, results in 32, images become smooth at 7534

# increasing learning rate for 64 to 0.001, results in 33

# increased pixel norm epsilon from 1e-8 to 1e-7, results in 36

# reducing learning rate for 64 to 0.0006, results in 37

# starting from run 37, 7214, lowering learning rate for 64 to 0.0003, results in 38

# lowering learning rate for 32 to 0.0003, and 64 to 0.00015, results in 39

# starting from run 16 6874 again, results in 40

# Lowering learning rate for 64 to 0.00008, results in 41

# Changing learning rate back to 0.0003 , changing Adam epsilon from 1e-7 to to 1e-5, results in 42

# Changing pixel norm epsilon to 1e-5, results in 43

# Changing generator learning rate to 0.001 for 64, results in 44

# Changing generator learning rate to 0.0005 for 64, results in 45

# Changing generator learning rate to 0.0003 for 64, D to 0.0006, results in 46, fails directly

# Changing generator learning rate to 0.0012 for 64, D to 0.0003, results in 47

# Changing learning rate back to 0.0006 for both G and D, added back minibatch std layer, results in 48

# Updating the generator 2 times per update, but discriminator only 1 time per update, results in 50

# Starting from run 16, learning rate 0.0003 for D 64, 0.00005 for G 64, results in 51

# Starting from run 16, learning rate 0.0003 for D 64, 0.0002 for G 64, results in 52

# Fixed fan in bug, changing learning rate to 0.0004 for D and G, results in 53

# changing learning rate to 0.0005 for G, results in 54

# changing learning rate to 0.0006 for G, results in 55

# starting from run 55, 7354, changing LR to 0.0003 and 0.0002, results in 56

# lowering LR to 0.0002 and 0.00015, results in 57

# lowering LR to 0.00015 and 0.0001, results in 58

# changing LR to 0.0003 for D and G, updating G 2 times per D update, results in 59

# Lowering LR to 0.00015, results in 60

# Lowering G LR to 0.0001, results in 61

#-----
# Starting from scratch

# Changed upsampling function back to tile version, results in 64

# Increasing batch size for 32 from 8 to 16, to half the number of updates, results in 65

# Increasing learning rate to 0.0005, results in 66

# Lowering batch size for 32 to 4, results in 67

# Increasing batch size for 32 to 32, learning rate to 0.0048 (0.0006 * 32/4), results in 68

# Lowering LR for 32 to 0.0017 (0.0006 * sqrt(32/4)), results in 69

# Going back to batch size 4, learning rate 0.0006, results in 70

# Changing batch size to 4 for 64, increasing LR from 0.0003 to 0.0006, results in 71

# Changing LR for 64 to 0.0007, results in 72

#desc += '-preset-v2-4gpus'; num_gpus = 4; sched.minibatch_base = 4; sched.minibatch_dict = {4: 512, 8: 256, 16: 16, 32: 4, 64: 4, 128: 4}; sched.G_lrate_dict = {4: 0.0003, 8: 0.0003, 16: 0.0006, 32: 0.0006, 64: 0.0007, 128: 0.0006}; sched.D_lrate_dict = EasyDict(sched.G_lrate_dict); sched.D_lrate_dict = {4: 0.0003, 8: 0.0003, 16: 0.0006, 32: 0.0006, 64: 0.0007, 128: 0.0006}  ; train.total_kimg = 20000

# Lowering LR for 32 from 0.0003 to 0.0001, results in 65 (could increase batch size to 16 instead, half as many updates?)

# Adaptive updating, only update discriminator if accuracy is < 0.8, results in 50

#-----

# Starting from scratch again

# increased pixel norm epsilon from 1e-8 to 1e-7, to prevent that magnitudes become too high

# results in 34

# restarting from run 34, 4854, lowering batch size back to 4 for 32 (could try increasing learning rate), results in 35

#desc += '-preset-v2-4gpus'; num_gpus = 4; sched.minibatch_base = 4; sched.minibatch_dict = {4: 512, 8: 256, 16: 16, 32: 4, 64: 16, 128: 8}; sched.G_lrate_dict = {4: 0.0003, 8: 0.0003, 16: 0.0006, 32: 0.0006, 64: 0.001, 128: 0.0003}; sched.D_lrate_dict = EasyDict(sched.G_lrate_dict); sched.D_lrate_dict = {4: 0.0003, 8: 0.0003, 16: 0.0006, 32: 0.0006, 64: 0.001, 128: 0.0003}  ; train.total_kimg = 20000



# put back minibatch standard deviation


# Numerical precision (choose one).
desc += '-fp32'; sched.max_minibatch_per_gpu = {256: 1, 512: 1, 1024: 1}
#desc += '-fp16'; G.dtype = 'float16'; D.dtype = 'float16'; G.pixelnorm_epsilon=1e-4; G_opt.use_loss_scaling = True; D_opt.use_loss_scaling = True; sched.max_minibatch_per_gpu = {128: 1, 256: 1, 512: 1, 1024: 1}

# Disable individual features.
#desc += '-nogrowing'; sched.lod_initial_resolution = 1024; sched.lod_training_kimg = 0; sched.lod_transition_kimg = 0; train.total_kimg = 10000
#desc += '-nopixelnorm'; G.use_pixelnorm = False
#desc += '-nowscale'; G.use_wscale = False; D.use_wscale = False
#desc += '-noleakyrelu'; G.use_leakyrelu = False
#desc += '-nosmoothing'; train.G_smoothing = 0.0
#desc += '-norepeat'; train.minibatch_repeats = 1
#desc += '-noreset'; train.reset_opt_for_new_lod = False

# Special modes.
#desc += '-BENCHMARK'; sched.lod_initial_resolution = 4; sched.lod_training_kimg = 3; sched.lod_transition_kimg = 3; train.total_kimg = (8*2+1)*3; sched.tick_kimg_base = 1; sched.tick_kimg_dict = {}; train.image_snapshot_ticks = 1000; train.network_snapshot_ticks = 1000
#desc += '-BENCHMARK0'; sched.lod_initial_resolution = 1024; train.total_kimg = 10; sched.tick_kimg_base = 1; sched.tick_kimg_dict = {}; train.image_snapshot_ticks = 1000; train.network_snapshot_ticks = 1000
#desc += '-VERBOSE'; sched.tick_kimg_base = 1; sched.tick_kimg_dict = {}; train.image_snapshot_ticks = 1; train.network_snapshot_ticks = 100
#desc += '-GRAPH'; train.save_tf_graph = True
#desc += '-HIST'; train.save_weight_histograms = True

#----------------------------------------------------------------------------
# Utility scripts.
# To run, uncomment the appropriate line and launch train.py.

#train = EasyDict(func='util_scripts.generate_fake_images', run_id=23, num_pngs=1000); num_gpus = 1; desc = 'fake-images-' + str(train.run_id)
#train = EasyDict(func='util_scripts.generate_fake_images', run_id=23, grid_size=[15,8], num_pngs=10, image_shrink=4); num_gpus = 1; desc = 'fake-grids-' + str(train.run_id)
#train = EasyDict(func='util_scripts.generate_interpolation_video', run_id=23, grid_size=[1,1], duration_sec=60.0, smoothing_sec=1.0); num_gpus = 1; desc = 'interpolation-video-' + str(train.run_id)
#train = EasyDict(func='util_scripts.generate_training_video', run_id=23, duration_sec=20.0); num_gpus = 1; desc = 'training-video-' + str(train.run_id)

#train = EasyDict(func='util_scripts.evaluate_metrics', run_id=23, log='metric-swd-16k.txt', metrics=['swd'], num_images=16384, real_passes=2); num_gpus = 1; desc = train.log.split('.')[0] + '-' + str(train.run_id)
#train = EasyDict(func='util_scripts.evaluate_metrics', run_id=23, log='metric-fid-10k.txt', metrics=['fid'], num_images=10000, real_passes=1); num_gpus = 1; desc = train.log.split('.')[0] + '-' + str(train.run_id)
#train = EasyDict(func='util_scripts.evaluate_metrics', run_id=23, log='metric-fid-50k.txt', metrics=['fid'], num_images=50000, real_passes=1); num_gpus = 1; desc = train.log.split('.')[0] + '-' + str(train.run_id)
#train = EasyDict(func='util_scripts.evaluate_metrics', run_id=23, log='metric-is-50k.txt', metrics=['is'], num_images=50000, real_passes=1); num_gpus = 1; desc = train.log.split('.')[0] + '-' + str(train.run_id)
#train = EasyDict(func='util_scripts.evaluate_metrics', run_id=23, log='metric-msssim-20k.txt', metrics=['msssim'], num_images=20000, real_passes=1); num_gpus = 1; desc = train.log.split('.')[0] + '-' + str(train.run_id)

#----------------------------------------------------------------------------