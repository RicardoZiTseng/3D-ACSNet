import sys
import os
from models.model import DenseUNet
from util.model_fusion import deploy
from util.misc import maybe_mkdir_p

def main(params):
    ori_model = params.ori_model
    save_fold = params.save_fold
    maybe_mkdir_p(os.path.abspath(save_fold))
    net = DenseUNet()
    model = net.build_model()
    net.switch_to_deploy()
    deploy_model = net.build_model()
    print("Total {} model(s) need to be fused.".format(len(ori_model)))
    for i in range(len(ori_model)):
        model_i_path = os.path.abspath(ori_model[i])
        print("Fusing model: {} ...".format(model_i_path))
        deploy_path = os.path.join(save_fold, model_i_path.split('/')[-1].split('.')[0]+'_deploy.h5')
        deploy(model, deploy_model, model_i_path, deploy_path, 1e-3)
        print("Fused model has been stored in {}".format(deploy_path))
