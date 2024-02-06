## Universal_Attack.py -- The main entry file for attack generation
##
## Copyright (C) 2018, IBM Corp
##                     PaiShun Ting <paishun@umich.edu>
##                     Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
##                     Sijia Liu <sijia.liu@ibm.com>
##                     Chun-Chen Tu <timtu@umich.edu>
## Copyright (C) 2017, Huan Zhang <ecezhang@ucdavis.edu>.
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

import sys

sys.path.append('models/')
sys.path.append('optimization_methods/')

import os
import numpy as np
import argparse

from setup_mnist import MNIST, MNISTModel
import Utils as util
import optimization_methods.ObjectiveFunc as ObjectiveFunc
import optimization_methods.FZCGS as fzcgs
import optimization_methods.SGFFW as sgffw

import tensorflow as tf

from SysManager import SYS_MANAGER

MGR = SYS_MANAGER()


def main():
    
    data, model =  MNIST(), MNISTModel(restore="models/mnist", use_log=True)
    origImgs, origLabels, origImgID = util.generate_attack_data_set(data, model, MGR)

    # Initialize the adversarial perturbation as a zero array of the same shape as the target images.
    delImgAT_Init = np.zeros(origImgs[0].shape)
    objfunc = ObjectiveFunc.OBJFUNC(MGR, model, origImgs, origLabels)

    #MGR.Add_Parameter('eta', MGR.parSet['alpha']/origImgs[0].size)
    MGR.Log_MetaData()


    ########### OPTIMIZER CHOICE ############

    if(MGR.parSet['optimizer'] == 'FZCGS'):
        delImgAT = fzcgs.FZCGS(delImgAT_Init, MGR.parSet['nStage'], MGR.parSet['q'], MGR.parSet['K'], MGR.parSet['L'], objfunc, MGR)

    elif(MGR.parSet['optimizer'] == 'SGFFW'):
        delImgAT = sgffw.SGFFW(delImgAT_Init, MGR.parSet['nStage'], MGR.parSet['m'], objfunc, MGR.parSet['grad_approx_scheme'], MGR)

    ########################################

    else:
        print('Please specify a valid optimizer')


    # After obtaining the adversarial perturbation, iterate over each image to generate and save adversarial examples.
    for idx_ImgID in range(MGR.parSet['nFunc']):
        currentID = origImgID[idx_ImgID]

        # Predict the original label for the current image.
        # np.expand_dims is used because TensorFlow expects the input to be batched. That is, even if we only predicting on a single image, 
        # the model expects it to be within a batch - a 4D array with dimensions representing [batch_size, height, width, channels]. 
        # np.expand_dims adds an additional dimension to origImgs[idx_ImgID], which is a single image, 
        # turning a 3D array into a 4D array with a batch size of 1.
        orig_prob = model.model.predict(np.expand_dims(origImgs[idx_ImgID], axis=0))

        # Apply the perturbation to the original image.
        # Usage of tanh and arctanh: 
        # - np.arctanh: This function is applied to the scaled original image origImgs[idx_ImgID]*1.9999999. 
        #               The scaling by 1.9999999 is done to ensure the pixel values are within the domain of the arctanh function, which is (-1, 1). 
        #               This is important because arctanh has singularities at -1 and 1.
        # - Adding delImgAT: The adversarial perturbation delImgAT is then added to this 'arctanh' space. 
        #                    Since the arctanh is an increasing function, adding a perturbation in this space corresponds to 
        #                    adding a perturbation to the original image intensities in a nonlinear way, which might be beneficial for the attack.
        # - np.tanh: After adding the perturbation, tanh is applied. tanh has a range of (-1, 1), 
        #            which maps the modified values back to valid image intensities.
        advImg = np.tanh(np.arctanh(origImgs[idx_ImgID]*1.9999999)+delImgAT)/2.0

        # Predict the probability of the adversarial image.
        adv_prob  = model.model.predict(np.expand_dims(advImg, axis=0))

        suffix = "id{}_Orig{}_Adv{}".format(currentID, np.argmax(orig_prob), np.argmax(adv_prob))
        util.save_img(advImg, "{}/Adv_{}.png".format(MGR.parSet['save_path'], suffix))
    util.save_img(np.tanh(delImgAT)/2.0, "{}/Delta.png".format(MGR.parSet['save_path']))

    sys.stdout.flush()
    MGR.logHandler.close()


if __name__ == "__main__":
    #tf.get_logger().setLevel('WARNING')
    #tf.keras.utils.disable_interactive_logging()
    parser = argparse.ArgumentParser()

    ##### GENERAL PARAMETERS #####
    parser.add_argument('-optimizer', default='FZCGS', help="choose from FZCGS and SGFFW")
    parser.add_argument('-nFunc', type=int, default=10, help="Number of images being attacked at once")
    parser.add_argument('-target_label', type=int, default=4, help="The target digit to attack")
    #parser.add_argument('-alpha', type=float, default=1.0, help="Optimizer's step size being (alpha)/(input image size)")
    #parser.add_argument('-M', type=int, default=50, help="Length of each stage/epoch")
    parser.add_argument('-nStage', type=int, default=1000, help="Number of stages/epochs")
    parser.add_argument('-const', type=float, default=3, help="Weight put on the attack loss")
    parser.add_argument('-batch_size', type=int, default=5, help="Number of functions sampled for each iteration in the optmization steps")
    parser.add_argument('-rv_dist', default='UnitSphere', help="Choose from UnitSphere and UnitBall")


    ##### FZCGS PARAMETERS #####
    parser.add_argument('-q', type=int, default=3, help="batch size for S2 in FZCGS")
    parser.add_argument('-K', type=float, default=0.1, help="K parameter for FZCGS")
    parser.add_argument('-L', type=float, default=50, help="L (Lipschitz constant) parameter for FZCGS")
    

    ##### SGFFW PARAMETERS ####
    parser.add_argument('-grad_approx_scheme', default='RDSA', help="Choose stochastic gradient approximation scheme between KWSA, RDSA and I-RDSA")
    parser.add_argument('-m', type = float, default=50, help="number of random vectors for I-RDSA approximation scheme (SGFFW)")


    args = vars(parser.parse_args())

    if args['optimizer'] == 'FZCGS':
        MGR.Add_Parameter('save_path', 'Results/' + args['optimizer'])
    else:
        MGR.Add_Parameter('save_path', 'Results/' + args['optimizer'] + '/' + args['grad_approx_scheme'])

    for par in args:
        MGR.Add_Parameter(par, args[par])


    MGR.parSet['batch_size'] = min(MGR.parSet['batch_size'], MGR.parSet['nFunc'])

    main()
