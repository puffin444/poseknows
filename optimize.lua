function optimize()

	local criterion = (nn.MSECriterion())
	criterion = criterion:cuda();

	local kldcrit = nn.KLDCriterion();
	kldcrit = kldcrit:cuda();

        local state = {  }
        local statevae = { }
        local statepred = {  }
        local statedec = { }

        local w, grad = dpt:getParameters();
        local wlstm, gradlstm = lstm:getParameters();
        local wlstm_decoder, gradlstm_decoder = lstm_decoder:getParameters();
        local wvae, gradvae = vae:getParameters();
	
	local theImages = torch.Tensor(opt.batchsize,3,240,320);
	local theRealPos = torch.Tensor(opt.batchsize,3,120,160);

	local theMag = torch.Tensor(opt.batchsize,2*opt.numSteps);
	local thePos = torch.Tensor(opt.batchsize,opt.numSteps, 3,18);
	local theChange = torch.Tensor(opt.batchsize,36,opt.numSteps);


	local theMagDec = torch.Tensor(opt.batchsize,2*opt.numDecSteps);
	local thePosDec = torch.Tensor(opt.batchsize,opt.numDecSteps, 3,18);
	local theChangeDec = torch.Tensor(opt.batchsize,36,opt.numDecSteps);

	local theHom = torch.Tensor(opt.batchsize,2,80,60);

	theImages = theImages:cuda();
	thePos = thePos:cuda();
	thePosDec = thePosDec:cuda();
	theChange = theChange:cuda();
	theChangeDec = theChangeDec:cuda();
	theHom = theHom:cuda();
	theMag = theMag:cuda();
	theMagDec = theMagDec:cuda();
	theRealPos = theRealPos:cuda();

	function evalpred()
		dpt:zeroGradParameters();
		lstm:zeroGradParameters();

		theChange = theChange:reshape(opt.batchsize, 36*opt.numSteps);
		thePosDec = thePosDec:reshape(opt.batchsize, opt.numDecSteps, 3*18);
	        theChangeDec = theChangeDec:reshape(opt.batchsize, opt.numDecSteps*36);

		theFeatures = dpt:forward({theInputs, thePosDec, theChangeDec, theMagDec});

		vae:zeroGradParameters();
		vaeOutput = vae:forward({(theChange), (theMag), dpt.modules[1].modules[45].hiddenOutput:clone()});
		theSamps = aSample:forward(vaeOutput);

		theChange = theChange:reshape(opt.batchsize, 36, opt.numSteps);

		theMag = theMag:reshape(opt.batchsize, opt.numSteps, 2);

		lstm.modules[13].hiddenInput = dpt.modules[1].modules[45].hiddenOutput:clone()
		lstm.modules[13].cellInput = dpt.modules[1].modules[45].cellOutput:clone()

		predictions = lstm:forward({theSamps, thePos});

		theMag = theMag:reshape(opt.batchsize, opt.numSteps*2);
		theChange = theChange:reshape(opt.batchsize, 36*opt.numSteps);

		errDir = criterion:forward(predictions[1], theChange);
		errMag = criterion:forward(predictions[2], theMag);

		predGrad = {};

		predGrad[1] = criterion:backward(predictions[1], theChange):clone();
		predGrad[2] = criterion:backward(predictions[2], theMag):clone();

		featureGrad = lstm:backward({theSamps, thePos}, predGrad);

		print({errMag,errDir});

		err = 1.0*errMag+1.0*errDir;

		return err,gradlstm
	end

	function evaldeclstm()

		lstm_decoder:zeroGradParameters();

		local outputDec = theChangeDec:clone();
		local outputMag = theMagDec:clone();

		outputDec = outputDec:reshape(opt.batchsize, 36, opt.numDecSteps);
		outputDec = outputDec:transpose(3,2);	

		local tempDec = outputDec:clone();
		local tempDec2 = outputMag:clone();
		local temp3 = torch.CudaTensor(opt.batchsize, 4*opt.numDecSteps);

		tempDec:zero();
		tempDec2:zero();
		
		for i = opt.numDecSteps,1,-1 do
			print(i)
			tempDec:narrow(2,i,1):add(outputDec:narrow(2,opt.numDecSteps-i+1,1));
			tempDec2:narrow(2,i,1):add(outputMag:narrow(2,opt.numDecSteps-i+1,1));
		end

		tempDec = tempDec:reshape(opt.batchsize, opt.numDecSteps, 36);

		tempDec = tempDec:transpose(3,2);

		lstm_decoder.modules[13].hiddenInput = dpt.modules[1].modules[45].hiddenOutput:clone()
		lstm_decoder.modules[13].cellInput = dpt.modules[1].modules[45].cellOutput:clone()

		decode = lstm_decoder:forward({temp3:clone():zero(), thePosDec:clone():zero()});

		tempDec2 = tempDec2:reshape(opt.batchsize, opt.numDecSteps*2);
		tempDec = tempDec:reshape(opt.batchsize, 36*opt.numDecSteps);

		errDir = criterion:forward(decode[1], tempDec);
		errMag = criterion:forward(decode[2], tempDec2);

		decGrad = {};

		decGrad[1] = criterion:backward(decode[1], tempDec):clone();
		decGrad[2] = criterion:backward(decode[2], tempDec2):clone();

		lstm_decoder:backward({theSamps:clone():zero(), thePosDec:clone():zero()}, decGrad);

		print({errMag,errDir});

		errdec = 1.0*errMag+1.0*errDir;

		return errdec, gradlstm_decoder
	end

	function evalfeats()

		dpt.modules[1].modules[45].gradHiddenOutput = (lstm_decoder.modules[13].gradHiddenInput:clone() 
				+ lstm.modules[13].gradHiddenInput:clone()+vaeGrad[3]:clone());

		dpt.modules[1].modules[45].gradCellOutput = (lstm_decoder.modules[13].gradCellInput:clone()
				+ lstm.modules[13].gradCellInput:clone());

		local dptGrad = dpt:backward({theInputs, thePosDec, theChangeDec, theMagDec}, theFeatures:zero());

		return err, grad
	end

	function evalvae()
		vae:zeroGradParameters();
		aSample:zeroGradParameters();

		local theSampGrad = aSample:backward(vaeOutput, featureGrad[1]);

		errkld = kldcrit:forward(vaeOutput[1], vaeOutput[2]);
		errfinal = (1.0 - opt.recPerc)*errkld + opt.recPerc*err;

		kldgrad = {};
		kldgrad[1] = (1.0 - opt.recPerc)*kldcrit:backward(vaeOutput[1], vaeOutput[2])[1] + opt.recPerc*theSampGrad[1];
		kldgrad[2] = (1.0 - opt.recPerc)*kldcrit:backward(vaeOutput[1], vaeOutput[2])[2] + opt.recPerc*theSampGrad[2];

		vaeGrad = vae:backward({theChange, theMag, dpt.modules[1].modules[45].hiddenOutput:clone()}, kldgrad);

		return errfinal, gradvae
	end

	for i=1,opt.totalIter do

		nClock = os.clock();

		theHom = theHom:reshape(opt.batchsize,2,80,60);
		theImages = theImages:reshape(opt.batchsize,3,240,320);
		theRealPos = theRealPos:reshape(opt.batchsize,3,120,160);

        	loadup(theImages, theHom, thePos, theChange, theRealPos,
			thePosDec, theChangeDec, theMag, theMagDec);

		local tempPos = UpSample2:forward(theRealPos):clone();
		
		theHom = theHom:reshape(opt.batchsize,2,80,60);
		theHom = theHom:cuda();

		local theHomBig = UpSample:forward(theHom);
		theHomBig = theHomBig:reshape(opt.batchsize,2,320,240);	

		theImages = theImages:reshape(opt.batchsize,3,240,320);

		theInputs = torch.cat(theImages, tempPos, 2);
		theInputs = theInputs:transpose(4,3);

		theInputs = fulltransform(theInputs, theHomBig);

		theInputs = theInputs:transpose(4,3);
		theInputs = theInputs:reshape(opt.batchsize,6,240,320);
		theInputs = theInputs:transpose(2,1);
		theInputs = theInputs/255.0;

		theInputs[1] = theInputs[1] - pixMean[1];
		theInputs[2] = theInputs[2] - pixMean[2];
		theInputs[3] = theInputs[3] - pixMean[3];

		theInputs[4] = theInputs[4] - pixMean[1];
		theInputs[5] = theInputs[5] - pixMean[2];
		theInputs[6] = theInputs[6] - pixMean[3];

		theInputs = theInputs:transpose(2,1);

		errkld = 0;
		errdec = 0;

		thePos = thePos:reshape(opt.batchsize, opt.numSteps, 3*18);

		optim.adam(evalpred,wlstm,{learningRate = opt.learningRate, beta1= opt.beta1},statepred);

	        optim.adam(evalvae,wvae,{learningRate = opt.learningRate, beta1= opt.beta1},statevae);

		optim.adam(evaldeclstm,wlstm_decoder,{learningRate = opt.learningRate, beta1= opt.beta1},statedec);
			
		optim.adam(evalfeats,w,{learningRate = opt.learningRate, beta1= opt.beta1},statefeat);

		thePos = thePos:reshape(opt.batchsize, opt.numSteps, 3, 18);

		theChange = theChange:reshape(opt.batchsize, 36, opt.numSteps);

		thePosDec = thePosDec:reshape(opt.batchsize, opt.numDecSteps, 3, 18);

		theChangeDec = theChangeDec:reshape(opt.batchsize, 36, opt.numDecSteps);

		if (i%opt.printItr == 0) then

			theInputs = theInputs:transpose(2,1);

			theInputs[1] = theInputs[1] + pixMean[1];
			theInputs[2] = theInputs[2] + pixMean[2];
			theInputs[3] = theInputs[3] + pixMean[3];

			theInputs = theInputs:transpose(2,1);

			makeVid(theChange:clone(), thePos:clone(), theMag:clone(), 
				theInputs:narrow(2,1,3):clone(),1, opt.numSteps);

			predictions[1] = predictions[1]:reshape(opt.batchsize, 36, opt.numSteps);

			makeVid(predictions[1]:clone(), thePos:clone(), predictions[2]:clone(),
				theInputs:narrow(2,1,3):clone(), 2, opt.numSteps);
			
			decode[1] = decode[1]:reshape(opt.batchsize, 36, opt.numDecSteps);
			decode[1] = decode[1]:transpose(3,2);

			local temp = decode[1]:clone();
			local temp2 = decode[2]:clone();

			temp:zero();
			temp2:zero();
		
			for i = opt.numDecSteps,1,-1 do
				temp:narrow(2,i,1):add(decode[1]:narrow(2,opt.numDecSteps-i+1,1));
				temp2:narrow(2,i,1):add(decode[2]:narrow(2,opt.numDecSteps-i+1,1));
			end

			temp = temp:reshape(opt.batchsize, opt.numDecSteps, 36);
			temp = temp:transpose(3,2);			

			makeVid(temp:clone(), thePosDec:clone(), temp2:clone(), 
				theInputs:narrow(2,1,3):clone(), 6, opt.numDecSteps);

			makeVid(theChangeDec:clone(), thePosDec:clone(), theMagDec:clone(), 
				theInputs:narrow(2,1,3):clone(), 7, opt.numDecSteps);
			
			theInputs = theInputs:transpose(2,1);

			theInputs[1] = theInputs[1] - pixMean[1];
			theInputs[2] = theInputs[2] - pixMean[2];
			theInputs[3] = theInputs[3] - pixMean[3];

			theInputs = theInputs:transpose(2,1);

			vaeOutput[1] = vaeOutput[1]:zero();
			vaeOutput[2] = vaeOutput[2]:zero();

			theSamps = aSample:forward(vaeOutput);

			thePos = thePos:reshape(opt.batchsize, opt.numSteps, 3*18);

			predictions = dpt:forward({theInputs, thePosDec, theChangeDec, theMagDec});


			lstm.modules[13].hiddenInput = dpt.modules[1].modules[45].hiddenOutput:clone()
			lstm.modules[13].cellInput = dpt.modules[1].modules[45].cellOutput:clone()

			local theResult = doIterative(thePos,theInputs,thePosDec,theChangeDec,theMagDec);

			makeVid(theResult[1]:clone(), theResult[2]:clone(), theResult[3]:clone(),
				 theInputs:narrow(2,1,3):clone(), 4, opt.numSteps);

		end

		if i%60000 == 0 then
			opt.recPerc = 0.9995;
		end

		if i%opt.saveItr == 0 then
			torch.save(opt.outDir .. "dpt_" .. tostring(i) .. ".dat", dpt);

			torch.save(opt.outDir .. "dec_" .. tostring(i) .. ".dat", lstm_decoder);

			torch.save(opt.outDir .. "vae_" .. tostring(i) .. ".dat", vae);

			torch.save(opt.outDir .. "lstm_" .. tostring(i) .. ".dat", lstm);

			torch.save(opt.outDir .. "opt_" .. tostring(i) .. ".dat", opt);
		end

		lossFile = io.open(opt.outDir .. "loss.txt", "a");
		print(string.format("Iteration %d ; Pred err = %f\n", i, err))
		lossFile:write(string.format("Iteration %d ; Pred err = %f\n", i, err))

		print(string.format("Iteration %d ; Dec err = %f\n", i, errdec))
		lossFile:write(string.format("Iteration %d ; Dec err = %f\n", i, errdec))

		print(string.format("Iteration %d ; KLD err = %f\n", i, errkld))
		lossFile:write(string.format("Iteration %d ; KLD err = %f\n", i, errkld))

		lossFile:close();

		print(string.format("load %.2f \n", os.clock() - nClock))

		end
end
