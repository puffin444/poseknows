Cairo = require 'oocairo'
signal = require 'signal'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'stn'
require 'rnn'
require 'nn'
require 'image'
require 'optim'
require 'signal'
require 'nngraph'
require 'lmdb'
require 'unsup'
torch = require 'torch'

torch.setnumthreads(8);
cutorch.setDevice(1);

dofile("paintPose.lua")
dofile("imwarp.lua")
dofile("optimizeGAN.lua")
dofile("loadLMDBGAN.lua")
dofile("opts.lua")
dofile("printVideo.lua")
dofile("buildModels.lua");

local db = lmdb.env{
        Path = opt.dbGANPath,
        Name = opt.dbGANName
}

db:open();
print(db:stat());

txn = db:txn(true);
cursor = txn:cursor();


goodCounts = torch.load(opt.dbGANPath .. "/" .. 'goodCounts.t7');
print(goodCounts:size())

warper = nn.BilinearSamplerBHWD();
warper = warper:cuda();

UpSample = nn.SpatialUpSamplingBilinear(2);
UpSample = UpSample:cuda();

theMasks = torch.CudaTensor(opt.batchsizeGAN*32,3,opt.width,opt.height);
theMasks:zero();
theMasks = theMasks + 1.0;

pixMean = torch.Tensor(3);
pixMean[1] = 0.407
pixMean[2] = 0.456
pixMean[3] = 0.482

net = nn.Sequential()

--Adapted from Vondrick et al. NIPS 2016

defineD();
net_video = defineGV_unet(6,3,32);

-- initialize the model
local function weights_init(m)
  local name = torch.type(m)
  if name:find('Convolution') then
    m.weight:normal(0.0, 0.01)
    m.bias:fill(0)
  elseif name:find('BatchNormalization') then
    if m.weight then m.weight:normal(1.0, 0.02) end
    if m.bias then m.bias:fill(0) end
  end
end

netD = netD:cuda();
netD:apply(weights_init)
netD = cudnn.convert(netD, cudnn)

net_video = net_video:cuda();
net_video:apply(weights_init)
net_video = cudnn.convert(net_video, cudnn)

gpus = torch.range(1,1):totable()

curCount = 1;

net_video = nn.DataParallelTable(1):add(net_video, gpus):cuda();
netD = nn.DataParallelTable(1):add(netD, gpus):cuda();

optimize()
		


