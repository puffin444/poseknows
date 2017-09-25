function testRun()

	
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

		thePos = thePos:reshape(opt.batchsize, opt.numSteps, 3*18);

		curExamp = i;
		evaltest(thePos, thePosDec, theChangeDec, theMagDec, theChange, theMag)
		err = 0;
		errkld = 0;
		errdec = 0;

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
