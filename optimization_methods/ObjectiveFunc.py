## ObjectiveFunc.py -- Perform Gradient Estimation and Evaluation for a Given Function
##
## Copyright (C) 2018, IBM Corp
##                     PaiShun Ting <paishun@umich.edu>
##                     Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
##                     Sijia Liu <sijia.liu@ibm.com>
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

import numpy as np
import Utils as util

np.random.seed(2018)


class OBJFUNC:

    def __init__(self, MGR, model, origImgs, origLabels):

        self.const = MGR.parSet['const']
        self.model = model
        self.origImgs = origImgs
        self.origImgsAT = np.arctanh(origImgs*1.9999999)
        self.origLabels = origLabels
        self.nFunc = origImgs.shape[0]
        self.imageSize = np.size(origImgs)/self.nFunc
        self.query_count = 0
        self.Loss_L2 = 1e10
        self.Loss_Attack = 1e10
        self.Loss_Overall = self.Loss_L2 + self.const*self.Loss_Attack

        if(MGR.parSet['rv_dist'] == 'UnitBall'):
            self.RV_Gen = self.Draw_UnitBall
        elif(MGR.parSet['rv_dist'] == 'UnitSphere'):
            self.RV_Gen = self.Draw_UnitSphere
        else:
            print('Please specify a valid distribution for random perturbation')


    def Draw_UnitBall(self):
        sample = np.random.uniform(-1.0, 1.0, size=self.origImgs[0].shape)
        return sample/np.linalg.norm(sample.flatten())

    def Draw_UnitSphere(self):
        sample = np.random.normal(0.0, 1.0, size=self.origImgs[0].shape)
        return sample/np.linalg.norm(sample.flatten())
    

    def evaluate(self, delImgAT, randBatchIdx, addQueryCount = True):

        # If no specific batch indices are provided, use all nFunc component funtions
        if( randBatchIdx.size == 0 ):
            randBatchIdx = np.arange(0, self.nFunc)
        batchSize = randBatchIdx.size

        ####################################################################

        #this objective function is the same as the one in the original repo. 
        #constraints are enforced in the actual optimization methods.

        #delImgAT = np.clip(delImgAT_infn, -s, s)

        ####################################################################

        origLabels_Batched = self.origLabels[randBatchIdx]

        # Prepare the perturbed images for all components, but only a batch will be used later
        delImgsAT = np.repeat(np.expand_dims(delImgAT, axis=0), self.nFunc, axis=0)

        # Apply the perturbation to the original images and scale to [0, 1] range using the tanh function
        advImgs = np.tanh(self.origImgsAT + delImgsAT)/2.0

        advImgs_Batched = advImgs[randBatchIdx]

        if(addQueryCount):
            self.query_count += batchSize

        # Predict the scores of the adversarial images with MNIST model
        Score_AdvImgs_Batched = self.model.model.predict(advImgs_Batched)

        
        # Score_TargetLab and Score_NonTargetLab are used to compute the loss components 
        # for the correct class and the most competitive incorrect class, respectively.

        # Compute the score for the target label by taking the dot product of the original labels
        # (one-hot encoded) with the predicted scores, and apply clipping to avoid log(0)
        Score_TargetLab = np.maximum(1e-20, np.sum(origLabels_Batched*Score_AdvImgs_Batched, 1))

        # Compute the score for the non-target label by finding the maximum predicted score for
        # non-target classes, and apply clipping as before
        Score_NonTargetLab = np.maximum(1e-20, np.amax((1-origLabels_Batched)*Score_AdvImgs_Batched - (origLabels_Batched*10000),1))



        # Calculate the adversarial loss as the maximum of zero and the difference between the
        # negative log score for non-target and target labels. 
        # This represents the logit loss for the adversarial attack component.
        # LOGIT LOSS: the cross-entropy loss is a loss function used in binary classification tasks. 
        # It measures the performance of a classification model whose output is a probability value between 0 and 1. 
        # Logit loss increases as the predicted probability diverges from the actual label. 
        # It's used when the model outputs a probability for the class label, rather than just the most likely class.
        self.Loss_Attack = np.amax(np.maximum(0.0, -np.log(Score_NonTargetLab) + np.log(Score_TargetLab) ) )

        # Compute the L2 loss as the mean squared difference between the adversarial and original
        # images, scaled by the image size. This represents the perturbation's magnitude.
        self.Loss_L2 = self.imageSize * np.mean(np.square(advImgs-self.origImgs)/2.0)

        # The overall loss is a weighted sum of the L2 loss and the adversarial attack loss,
        # with the const weighting parameter.
        self.Loss_Overall = self.Loss_L2 + self.const*self.Loss_Attack

        return self.Loss_Overall



    def gradient_estimation(self, delImgAT, mu, q, randBatchIdx = np.array([])):
        f = self.evaluate(delImgAT, randBatchIdx)
        grad_avg = np.zeros(delImgAT.shape)
        for q_idx in range(q):
            u_rand = self.RV_Gen()
            f_perturb = self.evaluate(delImgAT + mu*u_rand, randBatchIdx)
            grad_avg += (f_perturb-f)*u_rand
        return (delImgAT.size/mu)*(grad_avg/q)

    def print_current_loss(self):
        print('Loss_Overall: ', self.Loss_Overall, ' Loss_L2: ', self.Loss_L2, ' Loss_Attack: ', self.Loss_Attack)
