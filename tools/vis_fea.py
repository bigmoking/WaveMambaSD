import os
import cv2
import torch
import numpy as np
from mmengine import Config
from mmyolo.utils.misc import get_file_list
from mmyolo.utils.boxam_utils import BoxAMDetectorWrapper

from sklearn.decomposition import PCA


##############################################
# hook工具
##############################################
class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = {}

    def _hook(self, name):
        def fn(module, input, output):
            self.features[name] = output.detach().cpu()

        return fn

    def register_hooks(self):
        backbone = self.model.backbone
        print(self.model)
        # 🔥 这里必须和你backbone名字一致
        backbone.wavelet_group1.ori.register_forward_hook(self._hook('Origin'))
        backbone.wavelet_group1.mamba.register_forward_hook(self._hook('mm'))
        # backbone.wavelet_group1.lfss.register_forward_hook(self._hook('LFSS'))
        # backbone.wavelet_group1.hl_identity.register_forward_hook(self._hook('HL'))
        # backbone.wavelet_group1.lh_identity.register_forward_hook(self._hook('LH'))
        # backbone.wavelet_group1.hh_identity.register_forward_hook(self._hook('HH'))
        #
        # backbone.wavelet_group1.h_blk.hl_identity_d.register_forward_hook(self._hook('HLD'))
        # backbone.wavelet_group1.h_blk.lh_identity_d.register_forward_hook(self._hook('LHD'))
        # backbone.wavelet_group1.h_blk.hh_identity_d.register_forward_hook(self._hook('HHD'))
        # backbone.wavelet_group1.ll_mamba.register_forward_hook(self._hook('LL_Mamba'))


##############################################
# 可视化函数
##############################################
def feature_to_image(feat, out_size=(512, 512)):
    feat = feat[0].numpy()  # [C,H,W]
    C, H, W = feat.shape

    # ===== PCA降维 =====
    feat = feat.reshape(C, -1).T  # HW,C

    pca = PCA(n_components=1)
    fmap = pca.fit_transform(feat)
    fmap = fmap.reshape(H, W)

    # ===== normalize =====
    fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-6)
    fmap = (fmap * 255).astype(np.uint8)

    # ===== resize到原图大小 =====
    fmap = cv2.resize(fmap, out_size)

    # 伪彩色（更适合论文）
    fmap = cv2.applyColorMap(fmap, cv2.COLORMAP_JET)

    return fmap


##############################################
# 主函数
##############################################
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('img', help='image path or dir')
    parser.add_argument('config', help='config file')
    parser.add_argument('checkpoint', help='checkpoint')
    parser.add_argument('--out-dir', default='./vis_feature')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ===== build model =====
    cfg = Config.fromfile(args.config)
    model_wrapper = BoxAMDetectorWrapper(
        cfg, args.checkpoint, score_thr=0.3, device=args.device)

    model = model_wrapper.detector

    # ===== 注册hook =====
    extractor = FeatureExtractor(model)
    extractor.register_hooks()

    # ===== 读取图片 =====
    image_list, source_type = get_file_list(args.img)

    for image_path in image_list:
        print(f'Processing: {image_path}')

        image = cv2.imread(image_path)
        model_wrapper.set_input_data(image)

        # ===== forward =====
        with torch.no_grad():
            _ = model_wrapper()

        base = os.path.basename(image_path).split('.')[0]

        # ===== 保存原图 =====
        cv2.imwrite(f"{args.out_dir}/{base}_input.png", image)

        # ===== 保存特征 =====
        for name, feat in extractor.features.items():
            fmap = feature_to_image(feat, out_size=(image.shape[1], image.shape[0]))

            cv2.imwrite(f"{args.out_dir}/{base}_{name}.png", fmap)

        # ===== 拼图（论文用）=====
        merge_visualization(image, extractor.features, args.out_dir, base)

        extractor.features.clear()


##############################################
# 拼图（论文Figure）
##############################################
def merge_visualization(image, feats, out_dir, base):
    h, w = image.shape[:2]

    def get(name):
        if name not in feats:
            return np.zeros((h, w, 3), dtype=np.uint8)
        return feature_to_image(feats[name], (w, h))

    # LL = get('LL')
    ori = get('Origin')
    mm = get('mm')
    # lfss = get('LFSS')


    # HLD = get('HLD')
    # LHD = get('LHD')
    # HHD = get('HHD')
    # # LL_M = get('LL_Mamba')

    # resize原图
    img = cv2.resize(image, (w, h))

    # 拼图布局
    # top = np.hstack([img, LL, LL_M])
    # bottom = np.hstack([HL, LH, HH])

    # bottom = np.hstack([HLD, LHD, HHD])
    # top= np.hstack([HL, LH, HH])


    canvas = np.vstack([img,ori, mm])

    cv2.imwrite(f"{out_dir}/{base}_merge.png", canvas)


##############################################
if __name__ == '__main__':
    main()

#
# python
# vis_feature.py \
#     demo.jpg \
#     your_config.py \
#     your_model.pth \
#     - -out - dir
# vis_out
