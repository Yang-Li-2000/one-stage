# for customized attention
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.cnn.bricks.transformer import ATTENTION



@ATTENTION.register_module()
class MyMultiheadAttention(MultiheadAttention):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):

        # TODO: replace self.attn with the customized one
        super(MyMultiheadAttention, self).__init__(embed_dims,
                                                   num_heads,
                                                   attn_drop,
                                                   proj_drop,
                                                   dropout_layer=dropout_layer,
                                                   init_cfg=init_cfg,
                                                   batch_first=batch_first,
                                                   **kwargs)
        # if 'dropout' in kwargs:
        #     warnings.warn(
        #         'The arguments `dropout` in MultiheadAttention '
        #         'has been deprecated, now you can separately '
        #         'set `attn_drop`(float), proj_drop(float), '
        #         'and `dropout_layer`(dict) ', DeprecationWarning)
        #     attn_drop = kwargs['dropout']
        #     dropout_layer['drop_prob'] = kwargs.pop('dropout')
        # self.attn = MyNN_MultiheadAttention(embed_dims, num_heads, attn_drop, **kwargs)

