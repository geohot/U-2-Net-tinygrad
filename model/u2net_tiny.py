from tinygrad import Tensor
from tinygrad.nn import Conv2d, BatchNorm2d

class REBNCONV():
    def __init__(self, in_ch=3, out_ch=3,dirate=1):
        self.conv_s1 = Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate)
        self.bn_s1 = BatchNorm2d(out_ch)

    def __call__(self, x: Tensor) -> Tensor:
        return self.bn_s1(self.conv_s1(x)).relu()
    
# upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src:Tensor,tar:Tensor) -> Tensor:
    out = src.interpolate(tar.shape, mode="linear")
    return out

class RSU7():
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)
        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def __call__(self,x:Tensor):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = hx1.max_pool2d(2, stride=2, ceil_mode=True)

        hx2 = self.rebnconv2(hx)
        hx = hx2.max_pool2d(2, stride=2, ceil_mode=True)

        hx3 = self.rebnconv3(hx)
        hx = hx3.max_pool2d(2, stride=2, ceil_mode=True)

        hx4 = self.rebnconv4(hx)
        hx = hx4.max_pool2d(2, stride=2, ceil_mode=True)

        hx5 = self.rebnconv5(hx)
        hx = hx5.max_pool2d(2, stride=2, ceil_mode=True)

        hx6 = self.rebnconv6(hx)
        hx7 = self.rebnconv7(hx6)

        hx6d =  self.rebnconv6d(hx7.cat(hx6,dim=1))
        hx6dup = _upsample_like(hx6d,hx5)

        hx5d =  self.rebnconv5d(hx6dup.cat(hx5,dim=1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(hx5dup.cat(hx4,dim=1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(hx4dup.cat(hx3,dim=1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(hx3dup.cat(hx2,dim=1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(hx2dup.cat(hx1,dim=1))

        return hx1d + hxin
    
class RSU6():
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)
        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def __call__(self,x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = hx1.max_pool2d(2, stride=2, ceil_mode=True)

        hx2 = self.rebnconv2(hx)
        hx = hx2.max_pool2d(2, stride=2, ceil_mode=True)

        hx3 = self.rebnconv3(hx)
        hx = hx3.max_pool2d(2, stride=2, ceil_mode=True)

        hx4 = self.rebnconv4(hx)
        hx = hx4.max_pool2d(2, stride=2, ceil_mode=True)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d =  self.rebnconv5d(hx6.cat(hx5,dim=1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(hx5dup.cat(hx4,dim=1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(hx4dup.cat(hx3,dim=1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(hx3dup.cat(hx2,dim=1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(hx2dup.cat(hx1,dim=1))

        return hx1d + hxin
    
class RSU5():
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)
        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def __call__(self,x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = hx1.max_pool2d(2, stride=2, ceil_mode=True)

        hx2 = self.rebnconv2(hx)
        hx = hx2.max_pool2d(2, stride=2, ceil_mode=True)

        hx3 = self.rebnconv3(hx)
        hx = hx3.max_pool2d(2, stride=2, ceil_mode=True)

        hx4 = self.rebnconv4(hx)
        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(hx5.cat(hx4,dim=1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(hx4dup.cat(hx3,dim=1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(hx3dup.cat(hx2,dim=1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(hx2dup.cat(hx1,dim=1))

        return hx1d + hxin
    
class RSU4():
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)
        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def __call__(self,x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = hx1.max_pool2d(2, stride=2, ceil_mode=True)

        hx2 = self.rebnconv2(hx)
        hx = hx2.max_pool2d(2, stride=2, ceil_mode=True)

        hx3 = self.rebnconv3(hx)
        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(hx4.cat(hx3, dim=1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(hx3dup.cat(hx2,dim=1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(hx2dup.cat(hx1,dim=1))

        return hx1d + hxin

class RSU4F():
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)
        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)
        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def __call__(self,x):
        hx = x

        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(hx4.cat(hx3,dim=1))
        hx2d = self.rebnconv2d(hx3d.cat(hx2,dim=1))
        hx1d = self.rebnconv1d(hx2d.cat(hx1,dim=1))

        return hx1d + hxin
    
class U2NET():
    def __init__(self,in_ch=3,out_ch=1):
        self.stage1 = RSU7(in_ch,32,64)
        self.stage2 = RSU6(64,32,128)
        self.stage3 = RSU5(128,64,256)
        self.stage4 = RSU4(256,128,512)
        self.stage5 = RSU4F(512,256,512)
        self.stage6 = RSU4F(512,256,512)

        # decoder
        self.stage5d = RSU4F(1024,256,512)
        self.stage4d = RSU4(1024,128,256)
        self.stage3d = RSU5(512,64,128)
        self.stage2d = RSU6(256,32,64)
        self.stage1d = RSU7(128,16,64)

        self.side1 = Conv2d(64,out_ch,3,padding=1)
        self.side2 = Conv2d(64,out_ch,3,padding=1)
        self.side3 = Conv2d(128,out_ch,3,padding=1)
        self.side4 = Conv2d(256,out_ch,3,padding=1)
        self.side5 = Conv2d(512,out_ch,3,padding=1)
        self.side6 = Conv2d(512,out_ch,3,padding=1)

        self.outconv = Conv2d(6*out_ch,out_ch,1)

    def __call__(self,x):
        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = hx1.max_pool2d(2, stride=2, ceil_mode=True)

        #stage 2
        hx2 = self.stage2(hx)
        hx = hx2.max_pool2d(2, stride=2, ceil_mode=True)

        #stage 3
        hx3 = self.stage3(hx)
        hx = hx3.max_pool2d(2, stride=2, ceil_mode=True)

        #stage 4
        hx4 = self.stage4(hx)
        hx = hx4.max_pool2d(2, stride=2, ceil_mode=True)

        #stage 5
        hx5 = self.stage5(hx)
        hx = hx5.max_pool2d(2, stride=2, ceil_mode=True)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)

        #-------------------- decoder --------------------
        hx5d = self.stage5d(hx6up.cat(hx5,dim=1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(hx5dup.cat(hx4,dim=1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(hx4dup.cat(hx3,dim=1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(hx3dup.cat(hx2,dim=1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(hx2dup.cat(hx1,dim=1))


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        d0 = self.outconv(d1.cat(d2,dim=1).cat(d3,dim=1).cat(d4,dim=1).cat(d5,dim=1).cat(d6,dim=1))

        return d0.sigmoid(), d1.sigmoid(), d2.sigmoid(), d3.sigmoid(), d4.sigmoid(), d5.sigmoid(), d6.sigmoid()
