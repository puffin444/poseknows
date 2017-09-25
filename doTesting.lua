Cairo = require 'oocairo'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'stn'
require 'nn'
require 'image'
require 'optim'
require 'hdf5'
require 'nngraph'
require 'lmdb'
require 'unsup'
torch = require 'torch'
VAE = require 'VAE'
require 'KLDCriterion'
require 'Sampler'  
require 'distributions'

dofile("opts.lua")
dofile('paintPose.lua')
dofile("imwarp.lua")
dofile("optimize.lua")
dofile("loadLMDB.lua")
dofile("makeVid.lua")
dofile("doIterative.lua")
dofile("testRun.lua")
dofile("evaltest.lua")
dofile("getIntermediates.lua")
dofile("makeIntermediate.lua")
dofile("buildModels.lua")

cutorch.setDevice(1);
torch.setnumthreads(8);

curBatch = 0;

local db = lmdb.env{
	Path = opt.dbPath,
	Name = opt.dbName
}

db:open();
print(db:stat());

txn = db:txn(true);
cursor = txn:cursor();

gpus = torch.range(1,1):totable()

opt.testGen = true;

pixMean = torch.Tensor(3);
pixMean[1] = 0.407
pixMean[2] = 0.456
pixMean[3] = 0.482

UpSample = nn.SpatialUpSamplingBilinear(4);
UpSample = UpSample:cuda();

UpSample2 = nn.SpatialUpSamplingBilinear(2);
UpSample2 = UpSample2:cuda();

warper = nn.BilinearSamplerBHWD();
warper = warper:cuda();

theMasks = torch.CudaTensor(opt.batchsize*opt.seqLen*2,3,320,240);
theMasks:zero();
theMasks = theMasks + 1.0;

aSample = nn.Sampler();
aSample = aSample:cuda();

dpt = nn.DataParallelTable(1):add(features, gpus):cuda();

dofile('loadModel.lua')

slstm = buildDecoderSingle();
		
testRun();



