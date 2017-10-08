function buildEncoder()

	local featuresInput = nn.Identity()()
	local iPose = nn.Identity()()
	local iDiff = nn.Identity()()
	local iMag = nn.Identity()()

	local iPose2 = nn.Reshape(opt.batchsize*opt.numDecSteps, 54)(iPose);
	iPose2 = nn.Linear(54, 54)(iPose2);
	iPose2 = nn.BatchNormalization(54)(iPose2);

	local iDiff2 = nn.Reshape(opt.batchsize, 36, opt.numDecSteps)(iDiff);
	iDiff2 = nn.Transpose({3,2})(iDiff2);
	iDiff2 = nn.Reshape(opt.batchsize*opt.numDecSteps, 36)(iDiff2);
	iDiff2 = nn.Linear(36, 54)(iDiff2);
	iDiff2 = nn.BatchNormalization(54)(iDiff2);

	local iMag2 = nn.Reshape(opt.batchsize*opt.numDecSteps, 2)(iMag);
	iMag2 = nn.Linear(2, 54)(iMag2);
	iMag2 = nn.BatchNormalization(54)(iMag2);


	local poseInput = nn.JoinTable(2)({iPose2, iDiff2, iMag2});
	poseInput = nn.Linear(54+54+54, 1024)(poseInput);
	poseInput = nn.BatchNormalization(1024)(poseInput);
	poseInput = nn.Reshape(opt.batchsize, opt.numDecSteps, 1024)(poseInput)
	poseInput = nn.Transpose({2,1})(poseInput);

	local sampInput = nn.Identity()()

	local features = (cudnn.SpatialConvolution(6,32,7,7,4,4,2,2))(featuresInput);       
	features = (cudnn.ReLU(true))(features)
	features = ((cudnn.SpatialBatchNormalization(32)))(features)
	features = (cudnn.SpatialMaxPooling(3,3,2,2))(features)                 
	features = (cudnn.SpatialConvolution(32,128,5,5,1,1,2,2))(features)       
	features = (cudnn.ReLU(true))(features)
	features = ((cudnn.SpatialBatchNormalization(128)))(features)
	features = (cudnn.SpatialMaxPooling(3,3,2,2))(features)                   
	features = (cudnn.SpatialConvolution(128,192,3,3,1,1,1,1))(features)     
	features = (cudnn.ReLU(true))(features)
	features = ((cudnn.SpatialBatchNormalization(192)))(features)
	features = (cudnn.SpatialConvolution(192,192,3,3,1,1,1,1))(features)      
	features = (cudnn.ReLU(true))(features)
	features = ((cudnn.SpatialBatchNormalization(192)))(features)
	features = (cudnn.SpatialConvolution(192,128,3,3,1,1,1,1))(features)      
	features = (cudnn.ReLU(true))(features)
	features = ((cudnn.SpatialBatchNormalization(128)))(features)
	features = (cudnn.SpatialMaxPooling(3,3,2,2))(features)                  
	features = nn.View(6*9*128)(features);
	features = nn.Linear(6*9*128, 1024)(features);
	features = nn.BatchNormalization(1024)(features)
	features = nn.View(opt.batchsize, 1024)(features);
	features = nn.Replicate(opt.numDecSteps)(features);
	features = nn.JoinTable(3)({features, poseInput});

	features = cudnn.LSTM(2048, opt.hiddenSizeFeatures,opt.LSTMLayers)(features);

	features = nn.gModule({featuresInput, iPose, iDiff, iMag}, {features});

	return features:cuda();

end

function buildDecoderSingle()
	local MagInput2 = nn.Identity()();
	local poseInput2 = nn.Identity()();

	local poseInput = nn.Reshape(opt.batchsize, 54)(poseInput2);
	poseInput = nn.Linear(54, 54)(poseInput);
	poseInput = nn.BatchNormalization(54)(poseInput);
	poseInput = nn.Reshape(opt.batchsize, 1, 54)(poseInput)


	local featuresRepeat = nn.Reshape(opt.batchsize, 4)(MagInput2);
	featuresRepeat = nn.Linear(4, 54)(featuresRepeat);
	featuresRepeat = nn.BatchNormalization(54)(featuresRepeat);
	featuresRepeat = nn.Reshape(opt.batchsize, 1, 54)(featuresRepeat);

	featuresRepeat = nn.JoinTable(3)({featuresRepeat, poseInput});

	featuresRepeat = nn.Transpose({2,1})(featuresRepeat);
	featuresRepeat = cudnn.LSTM(108, opt.hiddenSizeFeatures,opt.LSTMLayers)(featuresRepeat);

	featuresRepeat = nn.Transpose({2,1})(featuresRepeat);
	featuresRepeat = nn.View(opt.batchsize, opt.hiddenSizeFeatures)(featuresRepeat);

	local changeOut = nn.Linear(opt.hiddenSizeFeatures, 36)(featuresRepeat)
	changeOut = nn.View(opt.batchsize, 1, 36)(changeOut);
	changeOut = nn.Transpose({3,2})(changeOut);
	changeOut = nn.View(opt.batchsize, 36)(changeOut);

	local magOut = nn.Linear(opt.hiddenSizeFeatures, 2)(featuresRepeat)
	magOut = nn.View(opt.batchsize, 1, 2)(magOut);
	magOut = nn.View(opt.batchsize, 2)(magOut);

	local slstm = nn.gModule({MagInput2, poseInput2}, {changeOut, magOut});

	slstm = slstm:cuda();
	return slstm;
end

function buildDecoder(theSteps)

	local MagInput2 = nn.Identity()();
	local poseInput2 = nn.Identity()();

	local poseInput = nn.Reshape(opt.batchsize*theSteps, 54)(poseInput2);

	poseInput = nn.Linear(54, 54)(poseInput);
	poseInput = nn.BatchNormalization(54)(poseInput);
	poseInput = nn.Reshape(opt.batchsize, theSteps, 54)(poseInput)

	local featuresRepeat = nn.Reshape(opt.batchsize*theSteps, 4)(MagInput2);
	featuresRepeat = nn.Linear(4, 54)(featuresRepeat);
	featuresRepeat = nn.BatchNormalization(54)(featuresRepeat);
	featuresRepeat = nn.Reshape(opt.batchsize, theSteps, 54)(featuresRepeat);

	featuresRepeat = nn.JoinTable(3)({featuresRepeat, poseInput});

	featuresRepeat = nn.Transpose({2,1})(featuresRepeat);
	featuresRepeat = cudnn.LSTM(108, opt.hiddenSizeFeatures,opt.LSTMLayers)(featuresRepeat);

	featuresRepeat = nn.Transpose({2,1})(featuresRepeat);
	featuresRepeat = nn.View(opt.batchsize*theSteps, opt.hiddenSizeFeatures)(featuresRepeat);

	local changeOut = nn.Linear(opt.hiddenSizeFeatures, 36)(featuresRepeat)
	changeOut = nn.View(opt.batchsize, theSteps, 36)(changeOut);
	changeOut = nn.Transpose({3,2})(changeOut);
	changeOut = nn.View(opt.batchsize, theSteps*36)(changeOut);

	local magOut = nn.Linear(opt.hiddenSizeFeatures, 2)(featuresRepeat)
	magOut = nn.View(opt.batchsize, theSteps, 2)(magOut);
	magOut = nn.View(opt.batchsize, theSteps*2)(magOut);

	local lstm = nn.gModule({MagInput2, poseInput2}, {changeOut, magOut});

	local lstm = lstm:cuda();
	return lstm;

end

function buildVAE()
	local diffInput2 = nn.Identity()()
	local diffInput = nn.Linear(36*opt.numSteps,36*opt.numSteps)(diffInput2)
	diffInput = nn.BatchNormalization(36*opt.numSteps)(diffInput)
	diffInput = nn.Replicate(10, 2)(diffInput)
	diffInput = nn.Reshape(opt.batchsize, 10*36*opt.numSteps)(diffInput)

	local hiddenState = nn.Identity()()
	local hiddenState2 = nn.Transpose({2,1})(hiddenState);
	hiddenState2 = nn.Reshape(opt.batchsize, opt.hiddenSize*opt.LSTMLayers)(hiddenState2);

	local magInput2 = nn.Identity()()
	local magInput = nn.Linear(2*opt.numSteps,2*opt.numSteps)(magInput2)
	magInput = nn.BatchNormalization(2*opt.numSteps)(magInput)
	magInput = nn.Replicate(100, 2)(magInput)
	magInput = nn.Reshape(opt.batchsize, 100*2*opt.numSteps)(magInput)

	local joinedPred = nn.JoinTable(2)({diffInput,magInput, hiddenState2});

	local vaePred = nn.gModule({diffInput2, magInput2, hiddenState}, {joinedPred});
	local vae = VAE.get_encoder(vaePred, 100*2*opt.numSteps+10*36*opt.numSteps+opt.hiddenSize*opt.LSTMLayers, opt.vaeHidden, opt.encoderSize)
	vae = vae:cuda();

	return vae;
end

-- Code Adapted from Vondrick et al. NIPS 2016 and Isola
-- et al. CVPR 2017

function defineD()
  netD = nn.Sequential()
  netD:add(nn.VolumetricConvolution(3,128, 4,4,4, 2,2,2, 1,1,1))
  netD:add(nn.LeakyReLU(0.2, true))
  netD:add(nn.VolumetricConvolution(128,256, 4,4,4, 2,2,2, 1,1,1))
  netD:add(nn.VolumetricBatchNormalization(256,1e-3)):add(nn.LeakyReLU(0.2, true))
  netD:add(nn.VolumetricConvolution(256,512, 4,4,4, 2,2,2, 1,1,1))
  netD:add(nn.VolumetricBatchNormalization(512,1e-3)):add(nn.LeakyReLU(0.2, true))
  netD:add(nn.VolumetricConvolution(512,1024, 4,4,4, 2,2,2, 1,1,1))
  netD:add(nn.VolumetricBatchNormalization(1024,1e-3)):add(nn.LeakyReLU(0.2, true))
  netD:add(nn.VolumetricConvolution(1024,2, 2,5,4, 1,1,1, 0,0,0))
  netD:add(nn.View(2):setNumInputDims(4))
end

function defineGV_unet(input_nc, output_nc, ngf)
  local netG = nil
  -- input is (nc) x 256 x 256
  local e1 = - nn.VolumetricConvolution(input_nc, ngf, 4, 4, 4, 2, 2, 2, 1, 1, 1)
  -- input is (ngf) x 128 x 128
  local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.VolumetricConvolution(ngf, ngf * 2, 4, 4, 4, 2, 2, 2, 1, 1, 1) - nn.VolumetricBatchNormalization(ngf * 2)
  -- input is (ngf * 2) x 64 x 64
  local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.VolumetricConvolution(ngf * 2, ngf * 4, 4, 4, 4, 2, 2, 2, 1, 1, 1) - nn.VolumetricBatchNormalization(ngf * 4)
  -- input is (ngf * 4) x 32 x 32
  local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.VolumetricConvolution(ngf * 4, ngf * 8, 4, 4, 4, 2, 2, 2, 1, 1, 1) - nn.VolumetricBatchNormalization(ngf * 8)
  -- input is (ngf * 8) x 16 x 16
  local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.VolumetricConvolution(ngf * 8, ngf * 8, 4, 6, 6, 2, 2, 2, 1, 1, 1) - nn.VolumetricBatchNormalization(ngf * 8)
  -- input is (ngf * 8) x 8 x 8
  local d3_ = e5 - nn.ReLU(true) - nn.VolumetricFullConvolution(ngf * 8, ngf * 8, 4, 7, 6, 2, 2, 2, 1, 1, 1) - nn.VolumetricBatchNormalization(ngf * 8) - nn.Dropout(0.5)
  -- input is (ngf * 8) x 8 x 8
  local d3 = {d3_,e4} - nn.JoinTable(2)
  local d4_ = d3 - nn.ReLU(true) - nn.VolumetricFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 4, 2, 2, 2, 1, 1, 1) - nn.VolumetricBatchNormalization(ngf * 4) - nn.Dropout(0.5)
  -- input is (ngf * 8) x 16 x 16
  local d4 = {d4_,e3} - nn.JoinTable(2)
  local d5_ = d4 - nn.ReLU(true) - nn.VolumetricFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 4, 2, 2, 2, 1, 1, 1) - nn.VolumetricBatchNormalization(ngf * 2)
  -- input is (ngf * 4) x 32 x 32
  local d5 = {d5_,e2} - nn.JoinTable(2)
  local d6_ = d5 - nn.ReLU(true) - nn.VolumetricFullConvolution(ngf * 2 * 2, ngf, 4, 4, 4, 2, 2, 2, 1, 1, 1) - nn.VolumetricBatchNormalization(ngf)
  -- input is (ngf * 2) x 64 x 64
  local d6 = {d6_,e1} - nn.JoinTable(2)
  local d7 = d6 - nn.ReLU(true) - nn.VolumetricFullConvolution(ngf * 2, output_nc, 4, 4, 4, 2, 2, 2, 1, 1, 1)

  local o1 = d7 - nn.Tanh()

  netG = nn.gModule({e1},{o1})

  return netG
end

