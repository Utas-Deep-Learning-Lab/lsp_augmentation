import scipy.io as io
import numpy as np
import imageio
import imgaug as ia
import imgaug.augmenters as iaa

mat_file = './joints.mat'
mat = io.loadmat(mat_file)
joints = mat['joints']
joints = np.moveaxis(joints, [0, 1, 2], [2, 1, 0])


def augmentation(index):
    global x, y, kpsoi, image_aug, kpsoi_aug
    img = 'im' + str(index).rjust(4, '0') + '.jpg'
    im = imageio.imread('lsp_images/' + img)

    # 随机选一种增强方式，1可以换成2，随机两种
    aug = iaa.SomeOf(1, [
        iaa.Rotate((-30, 30)),
        iaa.ScaleX((0.75, 1.25)),
        iaa.ScaleY((0.75, 1.25))
    ])

    x1, y1 = joints[index - 1, :, 0][0], joints[index - 1, :, 1][0]
    x2, y2 = joints[index - 1, :, 0][1], joints[index - 1, :, 1][1]
    x3, y3 = joints[index - 1, :, 0][2], joints[index - 1, :, 1][2]
    x4, y4 = joints[index - 1, :, 0][3], joints[index - 1, :, 1][3]
    x5, y5 = joints[index - 1, :, 0][4], joints[index - 1, :, 1][4]
    x6, y6 = joints[index - 1, :, 0][5], joints[index - 1, :, 1][5]
    x7, y7 = joints[index - 1, :, 0][6], joints[index - 1, :, 1][6]
    x8, y8 = joints[index - 1, :, 0][7], joints[index - 1, :, 1][7]
    x9, y9 = joints[index - 1, :, 0][8], joints[index - 1, :, 1][8]
    x10, y10 = joints[index - 1, :, 0][9], joints[index - 1, :, 1][9]
    x11, y11 = joints[index - 1, :, 0][10], joints[index - 1, :, 1][10]
    x12, y12 = joints[index - 1, :, 0][11], joints[index - 1, :, 1][11]
    x13, y13 = joints[index - 1, :, 0][12], joints[index - 1, :, 1][12]
    x14, y14 = joints[index - 1, :, 0][13], joints[index - 1, :, 1][13]

    keypoints = [ia.Keypoint(x1, y1),
                 ia.Keypoint(x2, y2),
                 ia.Keypoint(x3, y3),
                 ia.Keypoint(x4, y4),
                 ia.Keypoint(x5, y5),
                 ia.Keypoint(x6, y6),
                 ia.Keypoint(x7, y7),
                 ia.Keypoint(x8, y8),
                 ia.Keypoint(x9, y9),
                 ia.Keypoint(x10, y10),
                 ia.Keypoint(x11, y11),
                 ia.Keypoint(x12, y12),
                 ia.Keypoint(x13, y13),
                 ia.Keypoint(x14, y14), ]

    kpsoi = ia.KeypointsOnImage(keypoints, shape=im.shape)
    # 看原图
    # ia.imshow(kpsoi.draw_on_image(im))
    aug_det = aug.to_deterministic()
    image_aug = aug_det.augment_image(im)
    kpsoi_aug = aug_det.augment_keypoints(kpsoi)
    # 可视化
    # newimg = ia.KeypointsOnImage(image_aug, shape=im.shape)
    # image_with_kps = kpsoi_aug.draw_on_image(newimg)
    # ia.imshow(image_with_kps)
    return image_aug, kpsoi_aug

    # 保存标签
    # a = kpsoi.to_xy_array()
    # print(a)


# 生成图片1000张
for i in range(10):
    augmentation(i + 1)
    k = '%03d' % (i + 1)
    imageio.imsave('test/im2{}.jpg'.format(k), image_aug)
    a = kpsoi_aug.to_xy_array()
    print(a)
