function doIterative(thePos,theInputs,thePosDec,theChangeDec,theMagDec)

	if(not (#lstm.modules == #slstm.modules)) then

		print("lstm != slstm. SOMETHING IS WRONG.");

	end

	for i = 1,#lstm.modules do
		if(slstm.modules[i].weight) then
			slstm.modules[i].weight = lstm.modules[i].weight:clone();
		end

		if(slstm.modules[i].bias) then
			slstm.modules[i].bias = lstm.modules[i].bias:clone();
		end
	end

	local inputPose = torch.CudaTensor(opt.batchsize, opt.numSteps, 54);
	inputPose = inputPose:zero();
	
	thePos = thePos:reshape(opt.batchsize, opt.numSteps, 54);

	inputPose = inputPose:reshape(opt.batchsize, opt.numSteps, 54);

	inputPose:narrow(2,1,1):add(thePos:narrow(2,1,1):clone());

	local unNormdiff = torch.CudaTensor(opt.batchsize, opt.numSteps*36);
	unNormdiff = unNormdiff:zero();

	local unNormdiff = unNormdiff:reshape(opt.batchsize, opt.numSteps*36);
	local unNormmag = torch.CudaTensor(opt.batchsize, opt.numSteps*2);
	unNormmag = unNormmag:zero();
        
	local predictions = dpt:forward({theInputs, thePosDec, theChangeDec, theMagDec});

	local poutput = {};

	slstm.modules[13].hiddenInput = dpt.modules[1].modules[45].hiddenOutput:clone()
	slstm.modules[13].cellInput = dpt.modules[1].modules[45].cellOutput:clone()

	poutput = slstm:forward({theSamps:narrow(2,1,4), inputPose:narrow(2,1,1)})
	slstm.modules[13].hiddenInput = slstm.modules[13].hiddenOutput:clone()
	slstm.modules[13].cellInput = slstm.modules[13].cellOutput:clone()

	unNormdiff:narrow(2,1, 36):add(poutput[1]);
	unNormmag:narrow(2,1, 2):add(poutput[2]);

	local steps = opt.numSteps;
	opt.numSteps = 1;

	poutput[1] = poutput[1]:reshape(opt.batchsize, 36, 1);

	doScalingInv(inputPose:clone(), poutput[1], poutput[2], opt.numSteps);

	poutput[1] = torch.cat(poutput[1], torch.CudaTensor(opt.batchsize,18):zero(),2)

	inputPose:narrow(2,2,1):add(inputPose:narrow(2,1,1), poutput[1]);

	opt.numSteps = steps;

	for i = 2,(opt.numSteps - 1) do
		poutput = slstm:forward({theSamps:narrow(2,(i-1)*5+1,4), inputPose:narrow(2,i,1)})

		slstm.modules[13].hiddenInput = slstm.modules[13].hiddenOutput:clone();
		slstm.modules[13].cellInput = slstm.modules[13].cellOutput:clone();

		unNormdiff:narrow(2,(i-1)*36 + 1, 36):add(poutput[1]);
		unNormmag:narrow(2,(i-1)*2 + 1, 2):add(poutput[2]);

		local steps = opt.numSteps;
		opt.numSteps = 1;

		poutput[1] = poutput[1]:reshape(opt.batchsize, 36, 1);

		doScalingInv(thePos:narrow(2,1,1):clone(), poutput[1], poutput[2], opt.numSteps);
		poutput[1] = torch.cat(poutput[1], torch.CudaTensor(opt.batchsize,18):zero(),2)

		inputPose:narrow(2,i+1,1):add(inputPose:narrow(2,i,1), poutput[1]);
		opt.numSteps = steps;
		
	end


	local poutput = slstm:forward({theSamps:narrow(2,opt.numSteps,4), inputPose:narrow(2,opt.numSteps,1)})
	unNormdiff:narrow(2,(opt.numSteps-1)*36 + 1, 36):add(poutput[1]);
	unNormmag:narrow(2,(opt.numSteps-1)*2 + 1, 2):add(poutput[2]);


	unNormdiff = unNormdiff:reshape(opt.batchsize, opt.numSteps, 36);

	unNormdiff = unNormdiff:transpose(3,2);

	return {unNormdiff, inputPose, unNormmag};

end
