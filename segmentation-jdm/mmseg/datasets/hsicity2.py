import os.path as osp

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from mmseg.utils.logger import get_root_logger

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class HSICity2Dataset(CustomDataset):
    """HGICity2 dataset
    """

    # CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    #            'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    #            'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    #            'bicycle')
    #
    # PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    #            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    #            [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    #            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    #            [0, 80, 100], [0, 0, 230], [119, 11, 32]]

    CLASSES = ('sky','tree','building','trunk','road')
    # CLASSES = ('shading1','shading2','shading3','shading4','shading5','shading6','shading7','shading8')
    # PALETTE =[[194,19,19], [3,139,43], [109,232,248], [12,50,78],[100,102,102]]#B G R
    PALETTE =[[19,19,194], [43,139,3], [248,232,109], [78,50,12],[102,102,100]]#R G B !!!

    # PALETTE =[[53,119,181], [245,128,6], [67,159,36], [204,43,41],
 # [145,104,190], [135,86,75], [219,120,195], [0,0,255]]

    PALETTE_NIR = [ [0, 0, 0], [32, 32, 32], [64, 64, 64], [96, 96, 96],
               [128, 128, 128], [160, 160, 160], [192, 192, 192], [224, 224, 224]]
    def __init__(self,
                 img_suffix='.tif',#'.tif'
                 seg_map_suffix='.png',#'_gray.png'
                 ref_suffix='.npy',
                 hsi_suffix='.tif',#'.tif'
                 **kwargs):
        super(HSICity2Dataset, self).__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, ref_suffix=ref_suffix, **kwargs)
        self.hsi_suffix = hsi_suffix

    @staticmethod
    def _convert_to_label_id(result):
        """Convert trainId to id for cityscapes."""
        if isinstance(result, str):
            result = np.load(result)
        import cityscapesscripts.helpers.labels as CSLabels
        result_copy = result.copy()
        for trainId, label in CSLabels.trainId2label.items():
            result_copy[result == trainId] = label.id

        return result_copy

    def results2img(self, results, imgfile_prefix, to_label_id, indices=None):
        """Write the segmentation results to images.

        Args:
            results (list[ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission.
            indices (list[int], optional): Indices of input results,
                if not set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if indices is None:
            indices = list(range(len(self)))

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):
            if to_label_id:
                result = self._convert_to_label_id(result)
            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            output = Image.fromarray(result.astype(np.uint8)).convert('P')
            import cityscapesscripts.helpers.labels as CSLabels
            palette = np.zeros((len(CSLabels.id2label), 3), dtype=np.uint8)
            for label_id, label in CSLabels.id2label.items():
                palette[label_id] = label.color

            output.putpalette(palette)
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files

    def format_results(self,
                       results,
                       imgfile_prefix,
                       to_label_id=True,
                       indices=None):
        """Format the results into dir (standard format for Cityscapes
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
            indices (list[int], optional): Indices of input results,
                if not set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
                the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """
        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, to_label_id,
                                        indices)

        return result_files

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 imgfile_prefix=None):
        """Evaluation in Cityscapes/default protocol.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            imgfile_prefix (str | None): The prefix of output image file,
                for cityscapes evaluation only. It includes the file path and
                the prefix of filename, e.g., "a/b/prefix".
                If results are evaluated with cityscapes protocol, it would be
                the prefix of output png files. The output files would be
                png images under folder "a/b/prefix/xxx.png", where "xxx" is
                the image name of cityscapes. If not specified, a temp file
                will be created for evaluation.
                Default: None.

        Returns:
            dict[str, float]: Cityscapes/default metrics.
        """

        eval_results = dict()
        metrics = metric.copy() if isinstance(metric, list) else [metric]
        if 'cityscapes' in metrics:
            eval_results.update(
                self._evaluate_cityscapes(results, logger, imgfile_prefix))
            metrics.remove('cityscapes')
        if len(metrics) > 0:
            eval_results.update(
                super(HSICity2Dataset, self).evaluate(results, metrics, logger))

        return eval_results

    def _evaluate_cityscapes(self, results, logger, imgfile_prefix):
        """Evaluation in Cityscapes protocol.

        Args:
            results (list): Testing results of the dataset.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            imgfile_prefix (str | None): The prefix of output image file

        Returns:
            dict[str: float]: Cityscapes evaluation results.
        """
        try:
            import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as CSEval  # noqa
        except ImportError:
            raise ImportError('Please run "pip install cityscapesscripts" to '
                              'install cityscapesscripts first.')
        msg = 'Evaluating in Cityscapes style'
        if logger is None:
            msg = '\n' + msg
        print_log(msg, logger=logger)

        result_dir = imgfile_prefix

        eval_results = dict()
        print_log(f'Evaluating results under {result_dir} ...', logger=logger)

        CSEval.args.evalInstLevelScore = True
        CSEval.args.predictionPath = osp.abspath(result_dir)
        CSEval.args.evalPixelAccuracy = True
        CSEval.args.JSONOutput = False

        seg_map_list = []
        pred_list = []

        # when evaluating with official cityscapesscripts,
        # **_gtFine_labelIds.png is used
        for seg_map in mmcv.scandir(
                self.ann_dir, 'gtFine_labelIds.png', recursive=True):
            seg_map_list.append(osp.join(self.ann_dir, seg_map))
            pred_list.append(CSEval.getPrediction(CSEval.args, seg_map))

        eval_results.update(
            CSEval.evaluateImgLists(pred_list, seg_map_list, CSEval.args))

        return eval_results

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, ref_dir, ref_suffix,
                         split):
        img_infos = []
        # for img in mmcv.scandir(img_dir, '.tif'):
        for img in mmcv.scandir(img_dir, self.img_suffix):
            name = img[:-4]
            if 's' not in name:
                # img_info = {
                #     'name': name,
                #     'filename': f'rgb{name}{img_suffix}',
                #     'hsi': img,
                #     'ann': dict(seg_map=f'rgb{name}{seg_map_suffix}')
                # }
                img_info = {
                    'name': name,
                    'filename': f'{name}{img_suffix}',
                    'hsi': img,
                    'ann': dict(seg_map=f'{name}{seg_map_suffix}'),
                    'ref': dict(seg_map=f'{name}{ref_suffix}') # wyb1218

                }
                img_infos.append(img_info)
        img_info = sorted(img_infos, key=lambda x: x['name'])
        print_log(f'Loaded {len(img_info)} images', logger=get_root_logger())
        return img_infos

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        ref_info = self.get_ref_info(idx)
        hsi_info = self.get_hsi_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info, hsi_info=hsi_info, ref_info=ref_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def get_hsi_info(self, idx):
        return self.img_infos[idx]['hsi']
