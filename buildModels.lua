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
