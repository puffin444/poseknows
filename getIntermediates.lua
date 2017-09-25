function getIntermediates(tempChange, tempPos, tempMag)

	doScalingInv(tempPos, tempChange,tempMag, opt.numSteps)

	local theChunk = opt.timestep/opt.numSteps;

	local theMult = torch.CudaTensor(theChunk);

	for i = 1,theChunk do
		theMult[i] = (opt.numSteps/opt.timestep)*1.0*i;
	end	

	theMult = theMult:reshape(1,1,theChunk);

	theMult = torch.repeatTensor(theMult, opt.batchsize, 36, 1);

	local theDiffs = tempChange:narrow(3,1,1);

	local aPose = tempPos:narrow(2,1,1);

	aPose = torch.repeatTensor(aPose:transpose(3,2), 1,1,theChunk);

	theDiffs = torch.repeatTensor(theDiffs, 1,1,theChunk);

	theDiffs:cmul(theMult);

	local zeroJunk = torch.CudaTensor(opt.batchsize, 18, theChunk);
	zeroJunk:zero();

	theDiffs = torch.cat(theDiffs, zeroJunk:clone(), 2);

	theDiffs:add(aPose);
	
	for i = 2,opt.numSteps do

		local aDiffs = tempChange:narrow(3,i,1);

		aDiffs = torch.repeatTensor(aDiffs, 1,1,theChunk);

		aDiffs:cmul(theMult);

		local aPose = tempPos:narrow(2,i,1);

		aPose = torch.repeatTensor(aPose:transpose(3,2), 1,1,theChunk);

		aDiffs = torch.cat(aDiffs, zeroJunk:clone(), 2);

		aDiffs:add(aPose);
		
		theDiffs = torch.cat(theDiffs, aDiffs, 3);

	end

	return theDiffs:transpose(3,2);

end
