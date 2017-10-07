function preprocess()
  theHom = theHom:cuda();
  theHom = theHom:view(opt.batchsizeGAN,32*2,80,64);
  theBigHom = UpSample:forward(theHom);
  theBigHom = theHom:view(opt.batchsizeGAN,32,2,opt.width,opt.height);
  theHom = theHom:view(opt.batchsizeGAN,32,2,80,64);

  referenceHom = theBigHom:narrow(2,1,1):clone(); 
  reference = theImages:narrow(2,1,1):clone();
  referencePos = theImagesPos:narrow(2,1,1):clone();

  reference = reference:transpose(5,4);
  referencePos = referencePos:transpose(5,4);
  theImages = theImages:transpose(5,4);
  theImagesPos = theImagesPos:transpose(5,4);

  reference = reference:reshape(opt.batchsizeGAN,1,3,opt.width,opt.height);
  referencePos = referencePos:reshape(opt.batchsizeGAN,1,3,opt.width,opt.height);
  referenceHom = referenceHom:reshape(opt.batchsizeGAN,1,2,opt.width,opt.height);

  reference = torch.repeatTensor(reference, 1, 32, 1, 1, 1);
  referencePos = torch.repeatTensor(referencePos, 1, 32, 1, 1, 1);
  referenceHom = torch.repeatTensor(referenceHom, 1, 32, 1, 1, 1);

  reference = reference:reshape(32*opt.batchsizeGAN,3,opt.width,opt.height);
  referencePos = referencePos:reshape(32*opt.batchsizeGAN,3,opt.width,opt.height);
  referenceHom = referenceHom:reshape(32*opt.batchsizeGAN,2,opt.width,opt.height);

  theImages = theImages:reshape(opt.batchsizeGAN*32,3,opt.width,opt.height);
  theBigHom = theBigHom:reshape(opt.batchsizeGAN*32,2,opt.width,opt.height);
  theImagesPos = theImagesPos:reshape(opt.batchsizeGAN*32,3,opt.width,opt.height);

  theMasks = theMasks:zero() + 1.0;

  theImages = theImages:transpose(4,2);
  theImagesPos = theImagesPos:transpose(4,2);
  theBigHom = theBigHom:transpose(4,2);
  theMasks = theMasks:transpose(4,2);
  reference = reference:transpose(4,2);
  referenceHom = referenceHom:transpose(4,2);
  referencePos = referencePos:transpose(4,2);

  theImages = theImages:clone();
  theImagesPos = theImagesPos:clone();
  theBigHom = theBigHom:clone();
  theMasks = theMasks:clone();
  reference = reference:clone();
  referenceHom = referenceHom:clone();
  referencePos = referencePos:clone();

  theImages = warper:forward({theImages, theBigHom}):clone();
  theImagesPos = warper:forward({theImagesPos, theBigHom}):clone();
  theMasks = warper:forward({theMasks, theBigHom}):clone();
  reference = warper:forward({reference, referenceHom}):clone();
  referencePos = warper:forward({referencePos, referenceHom}):clone();
  referencePos = referencePos:zero();

  theImages = theImages:transpose(4,2);
  theImagesPos = theImagesPos:transpose(4,2);
  theBigHom = theBigHom:transpose(4,2);

  theMasks = theMasks:transpose(4,2);
  reference = reference:transpose(4,2);
  referenceHom = referenceHom:transpose(4,2);
  referencePos = referencePos:transpose(4,2);

  theMasks[torch.le(theMasks, 0.0)] = 4.0;
  theMasks[torch.le(theMasks, 2.0)] = 0.0;
  theMasks[torch.ge(theMasks, 2.0)] = 1.0;

  theImages = torch.cmul(reference, theMasks) + theImages;

  K = nil;
  theBigHom = nil;
  referenceHom = nil;

  theImages[theImages:ne(theImages)] = 0;

  theImagesPos = torch.cmul(referencePos, theMasks) + theImagesPos;

  K = nil;
  referencePos = nil;
  theBigHom = nil;
  referenceHom = nil;

  collectgarbage();

  theImagesPos[theImagesPos:ne(theImagesPos)] = 0;

  theImages = theImages:transpose(4,3);
  theImagesPos = theImagesPos:transpose(4,3);

  theImages = theImages:reshape(opt.batchsizeGAN, 32, 3, opt.height, opt.width);

  theImages = theImages:reshape(opt.batchsizeGAN, 32, 3, opt.height, opt.width);

  theImagesPos = theImagesPos:reshape(opt.batchsizeGAN, 32, 3, opt.height, opt.width);

  reference = theImages:narrow(2,1,1):clone();
  reference = torch.repeatTensor(reference, 1, 32, 1, 1, 1);

  theInput = torch.concat({theImagesPos, reference}, 3);

  theImages = theImages:transpose(3,1);
  theInput = theInput:transpose(3,1);
  theImagesPos = theImagesPos:transpose(3,1);

  theImages[1] = theImages[1] - pixMean[1];
  theImages[2] = theImages[2] - pixMean[2];
  theImages[3] = theImages[3] - pixMean[3];

  theImagesPos[1] = theImagesPos[1] - pixMean[1];
  theImagesPos[2] = theImagesPos[2] - pixMean[2];
  theImagesPos[3] = theImagesPos[3] - pixMean[3];

  theInput[1] = theInput[1] - pixMean[1];
  theInput[2] = theInput[2] - pixMean[2];
  theInput[3] = theInput[3] - pixMean[3];
  theInput[4] = theInput[4] - pixMean[1];
  theInput[5] = theInput[5] - pixMean[2];
  theInput[6] = theInput[6] - pixMean[3];
 
  theImages = theImages:transpose(3,1);
  theImagesPos = theImagesPos:transpose(3,1);
  theInput = theInput:transpose(3,1);

  theImages = theImages:transpose(3,2);
  theImagesPos = theImagesPos:transpose(3,2);
  theInput = theInput:transpose(3,2);

end

function discriminator()
  gradParametersD:zero()
  label:fill(real_label)

  -- forward/backwards real examples
  outputR = netD:forward(theImages):clone()
  errD = criterion:forward(outputR, label)
  local df_do = criterion:backward(outputR, label):clone()
  netD:backward(theImages, df_do)

  -- generate fake examples
  fake = net_video:forward(theInput);
  label:fill(fake_label)

  -- forward/backwards fake examples
  output = netD:forward(fake):clone()
  errD = errD + criterion:forward(output, label)
  df_do = criterion:backward(output, label):clone()
  netD:backward(fake, df_do)

  errD = errD / 2

  netD:syncParameters();

  return errD, gradParametersD
end

function testgenerator()
  gradParametersG:zero()
  fake = net_video:forward(theInput);

  errReg = criterionReg:forward(fake, theImages) * opt.lambda
end

function generator()
  gradParametersG:zero()

  label:fill(real_label)
  err = criterion:forward(output, label)
  local df_do = criterion:backward(output, label):clone()
  local df_dg = netD:updateGradInput(fake, df_do):clone()

  errReg = criterionReg:forward(fake, theImages) * opt.lambda
  local df_reg = criterionReg:backward(fake, theImages) * opt.lambda

  net_video:backward(theInput, df_dg+df_reg)

  net_video:syncParameters();

  return err + errReg, gradParametersG
end


function optimize()

        state = { learningRate = 0.0002, beta1=0.5  }
        state2 = {  learningRate = 0.0002, beta1=0.5}

        --Adapted from Vondrick et al. NIPS 2016
 	criterion = nn.CrossEntropyCriterion()
	criterionReg = nn.AbsCriterion()
	criterion = criterion:cuda();
	criterionReg = criterionReg:cuda();


 	parametersD, gradParametersD = netD:getParameters()
 	parametersG, gradParametersG = net_video:getParameters()

 	real_label = 1
	fake_label = 2
 	err, errD, errReg = 0;

	label = torch.Tensor(opt.batchsizeGAN);
	theImages = torch.Tensor(opt.batchsizeGAN,32,3,opt.height,opt.width);
	theImagesPos = torch.Tensor(opt.batchsizeGAN,32,3,opt.height,opt.width);
	theHom = torch.Tensor(opt.batchsizeGAN,32,2,80,64);

	theImages = theImages:cuda();
	theImagesPos = theImagesPos:cuda();
	theHom = theHom:cuda();
	label = label:cuda();
	errD = 0;
	
	for i=1,opt.totalIterGAN do
		nClock = os.clock();
        	loadup(theImages, theImagesPos, theHom);

                preprocess()
    		
		optim.adam(discriminator, parametersD, state)
    		optim.adam(generator, parametersG, state2)
		
		theImages = theImages:transpose(3,2);
		theImagesPos = theImagesPos:transpose(3,2);
		theInput = theInput:transpose(3,2);
		fake = fake:transpose(3,2);

		if(i%opt.printItrGAN == 0) then

		  theImages = theImages:transpose(3,1);
		  fake = fake:transpose(3,1);
		  theInput = theInput:transpose(3,1);

		  theImages[1] = theImages[1] + pixMean[1];
		  theImages[2] = theImages[2] + pixMean[2];
		  theImages[3] = theImages[3] + pixMean[3];

		  fake[1] = fake[1] + pixMean[1];
		  fake[2] = fake[2] + pixMean[2];
		  fake[3] = fake[3] + pixMean[3];

		  theInput[1] = theInput[1] + pixMean[1];
		  theInput[2] = theInput[2] + pixMean[2];
		  theInput[3] = theInput[3] + pixMean[3];

		  theInput[4] = theInput[4] + pixMean[1];
		  theInput[5] = theInput[5] + pixMean[2];
		  theInput[6] = theInput[6] + pixMean[3];

		  theImages = theImages:transpose(3,1);
		  fake = fake:transpose(3,1);
		  theInput = theInput:transpose(3,1);

		  printVideo(theImages[1], 1,0);
		  printVideo(fake[1], 2,0);
		
                end

		if i%opt.saveItrGAN == 0 then
			torch.save(opt.outDirGAN .. "dis_" .. tostring(i) .. ".dat", netD);
			torch.save(opt.outDirGAN .. "gen_" .. tostring(i) .. ".dat", net_video);
			torch.save(opt.outDirGAN .. "opt_" .. tostring(i) .. ".dat", opt);
		end

		lossFile = io.open(opt.outDirGAN .. "loss.txt", "a");
		print(string.format("Iteration %d ; Dis err = %f\n", i, errD))
		lossFile:write(string.format("Iteration %d ; Dec err = %f\n", i, errD))

		print(string.format("Iteration %d ; Gen err = %f\n", i, err))
		lossFile:write(string.format("Iteration %d ; Pred err = %f\n", i, err))

		print(string.format("Iteration %d ; Pixel err = %f\n", i, errReg))
		lossFile:write(string.format("Iteration %d ; KLD err = %f\n", i, errReg))

		lossFile:close();

		print(string.format("load %.2f \n", os.clock() - nClock))
		
		print(os.clock() - nClock)

	end

end
