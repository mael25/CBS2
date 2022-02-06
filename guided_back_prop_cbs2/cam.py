import argparse
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import CBS2.cbs2.bird_view.models.image_gradcam as LBSimage
import myLBS.bird_view.utils.bz_utils as bzu


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--ppm', action='store_true', default=False,
                            help='Use PPM model')
    parser.add_argument('--fpn', action='store_true', default=False,
                            help='Use FPN model')
    parser.add_argument('--all', action='store_true', default=False,
                            help='All models')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    #####################################################################
    ## MODIF 15oct
    ##

    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'backbone': 'resnet34'
    }

    student_ppm = LBSimage.ImagePolicyModelSS(
                  config['backbone'],
                  ppm_bins=[1, 2, 3, 6]
                  ).to(config['device'])
    ckpt_ppm = '/storage2/mwildi/CBS2/cbs2/reference/ref_stud_ppm_l2/model-90.th'
    student_ppm.load_state_dict(torch.load(ckpt_ppm))
    # target_layers_ppm = [model.conv.layer4[-1]]

    student_fpn = LBSimage.ImagePolicyModelSS(
                  config['backbone'],
                  fpn=True
                  ).to(config['device'])
    ckpt_fpn = '/storage2/mwildi/CBS2/cbs2/reference/ref_stud_fpn_l2/model-100.th'
    student_fpn.load_state_dict(torch.load(ckpt_fpn))
    # target_layers_fpn = [model.fpn.downsamplers[-1]]

    student_orig = LBSimage.ImagePolicyModelSS(
                  config['backbone']
                  ).to(config['device'])
    ckpt_orig = '/storage2/mwildi/CBS2/cbs2/reference/ref_stud_original_l2/model-86.th'
    student_orig.load_state_dict(torch.load(ckpt_orig))
    # target_layers_orig = [model.conv.layer4[-1]]

    #model = models.resnet50(pretrained=True)


    if args.ppm:
        model = student_ppm
        # target_layers = target_layers_ppm
    if args.fpn:
        model = student_fpn
        # target_layers = target_layers_fpn
    else:
        model = student_orig
        # target_layers = target_layers_orig

    print(model)
    #####################################################################
    #model = models.resnet50(pretrained=True)

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])

    #target_layers = [model.layer4]


    image_names = [x[2] for x in os.walk(currentdir + os.path.sep + 'ins')][0]
    image_names = [x.split(".")[0] for x in image_names]
    img_array = []
    for image_name in sorted(image_names, key=int):

        image_path = 'ins' + os.path.sep + image_name + '.png'
        print(image_path)

        rgb_img_cv2 = cv2.imread(image_path, 1)[:, :, ::-1]
        rgb_img = np.float32(rgb_img_cv2) / 255
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])


        # We have to specify the target we want to generate
        # the Class Activation Maps for.
        # If targets is None, the highest scoring category (for every member in the batch) will be used.
        # You can target specific categories by
        # targets = [e.g ClassifierOutputTarget(281)]
        #targets = 0

        # Using the with statement ensures the context is freed, and you can
        # recreate different CAM objects in a loop.
        # cam_algorithm = methods[args.method]
        # with cam_algorithm(model=model,
        #                    target_layers=target_layers,
        #                    use_cuda=args.use_cuda) as cam:
        #
        #     # AblationCAM and ScoreCAM have batched implementations.
        #     # You can override the internal batch size for faster computation.
        #     cam.batch_size = 32
        #     grayscale_cam = cam(input_tensor=input_tensor,
        #                         targets=targets,
        #                         aug_smooth=args.aug_smooth,
        #                         eigen_smooth=args.eigen_smooth)
        #
        #     # Here grayscale_cam has only one image in the batch
        #     grayscale_cam = grayscale_cam[0, :]
        #
        #     cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        #
        #     # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        #     cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)



        if args.all:
            font = cv2.FONT_HERSHEY_SIMPLEX

            gb_model_ppm = GuidedBackpropReLUModel(model=student_ppm, use_cuda=args.use_cuda)
            gb_ppm = gb_model_ppm(input_tensor, target_category=None)
            gb_ppm = deprocess_image(gb_ppm)
            gb_ppm_viz = cv2.putText(gb_ppm.copy(), 'PPM', (10,10), font, 0.3, (255, 255, 255), 1, cv2.LINE_8)

            gb_model_fpn = GuidedBackpropReLUModel(model=student_fpn, use_cuda=args.use_cuda)
            gb_fpn = gb_model_fpn(input_tensor, target_category=None)
            gb_fpn = deprocess_image(gb_fpn)
            gb_fpn_viz = cv2.putText(gb_fpn.copy(), 'FPN', (10,10), font, 0.3, (255, 255, 255), 1, cv2.LINE_8)

            gb_model_orig = GuidedBackpropReLUModel(model=student_orig, use_cuda=args.use_cuda)
            gb_orig = gb_model_orig(input_tensor, target_category=None)
            gb_orig = deprocess_image(gb_orig)
            gb_orig_viz = cv2.putText(gb_orig.copy(), 'Original', (10,10), font, 0.3, (255, 255, 255), 1, cv2.LINE_8)

            viz_top = np.hstack((cv2.cvtColor(rgb_img_cv2, cv2.COLOR_BGR2RGB), gb_orig_viz))
            viz_bot = np.hstack((gb_ppm_viz, gb_fpn_viz))
            viz = np.vstack((viz_top, viz_bot))

            img_array.append(viz)

            #cv2.imwrite(f'outs/viz_{img_name}.jpg', viz)

        else:
            gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
            gb = gb_model(input_tensor, target_category=None)
            # cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
            # cam_gb = deprocess_image(cam_mask * gb)
            gb = deprocess_image(gb)


            #cv2.imwrite(f'outs/rgb{image_name}_{args.method}.jpg', cv2.cvtColor(rgb_img_cv2, cv2.COLOR_BGR2RGB))
            #cv2.imwrite(f'outs/{image_name}_{args.method}_cam.jpg', cam_image)
            #cv2.imwrite(f'outs/{image_name}_{args.method}_gb.jpg', gb)
            #cv2.imwrite(f'outs/{image_name}_{args.method}_cam_gb.jpg', cam_gb)

    if args.all:
        print(img_array[0].shape[:2])
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('outs/viz.avi',fourcc, 8, (img_array[0].shape[1], img_array[0].shape[0]))
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
