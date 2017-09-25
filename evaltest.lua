function evaltest(thePos, thePosDec, theChangeDec, theMagDec,theChange, theMag)
		vaeOutput = {};
		vaeOutput[1] = torch.CudaTensor(opt.batchsize, opt.encoderSize)
		vaeOutput[2] = torch.CudaTensor(opt.batchsize, opt.encoderSize)

		vaeOutput[1] = vaeOutput[1]:zero()
		vaeOutput[2] = vaeOutput[2]:zero()

		theSamps = aSample:forward(vaeOutput);

		thePos = thePos:reshape(opt.batchsize, opt.numSteps, 3*18);
	
		predictions = dpt:forward({theInputs, thePosDec, theChangeDec, theMagDec});

		lstm.modules[13].hiddenInput = dpt.modules[1].modules[45].hiddenOutput:clone()
		lstm.modules[13].cellInput = dpt.modules[1].modules[45].cellOutput:clone()

		local theResult = doIterative(thePos,theInputs,thePosDec,theChangeDec,theMagDec);

		theResult[1] = theResult[1]:reshape(1, opt.batchsize, 36, opt.numSteps);
		theResult[2] = theResult[2]:reshape(1, opt.batchsize, opt.numSteps, 54);
		theResult[3] = theResult[3]:reshape(1, opt.batchsize, 2*opt.numSteps);

		theInputs = theInputs:transpose(2,1);

		theInputs[1] = theInputs[1] + pixMean[1];
		theInputs[2] = theInputs[2] + pixMean[2];
		theInputs[3] = theInputs[3] + pixMean[3];

		theInputs = theInputs:transpose(2,1);

		for i = 1,(opt.numSamp - 1) do

			theSamps = aSample:forward(vaeOutput);

			local theResultTemp = doIterative(thePos,theInputs,thePosDec,theChangeDec,theMagDec);

			theResult[1] = torch.cat(theResult[1], theResultTemp[1]:reshape(1, opt.batchsize, 36, opt.numSteps), 1);
			theResult[2] = torch.cat(theResult[2], theResultTemp[2]:reshape(1, opt.batchsize, opt.numSteps, 54), 1);
			theResult[3] = torch.cat(theResult[3], theResultTemp[3]:reshape(1, opt.batchsize, 2*opt.numSteps), 1);
		end
                
                if(opt.numEval) then

                        batch = opt.batchsize;

                        theResult[1] = theResult[1]:reshape(opt.batchsize*opt.numSamp, 36, opt.numSteps);
                        theResult[2] = theResult[2]:reshape(opt.batchsize*opt.numSamp, opt.numSteps, 54);
                        theResult[3] = theResult[3]:reshape(opt.batchsize*opt.numSamp, opt.numSteps*2);

                        opt.batchsize = opt.batchsize*opt.numSamp;

                        doScalingInv(theResult[2], theResult[1],theResult[3], opt.numSteps)

                        theSamples = theResult[1]:clone();

                        opt.batchsize = batch;

                        theSamples = theSamples:reshape(opt.numSamp, opt.batchsize, 36*opt.numSteps);

                        doScalingInv(thePos, theChange,theMag, opt.numSteps)

                        theGT = theChange:clone();
                        theGT = theGT:reshape(opt.batchsize,36*opt.numSteps);
                        theGT = theGT:reshape(1,opt.batchsize, 36*opt.numSteps);
                        theGT = torch.repeatTensor(theGT, opt.numSamp, 1, 1);
                        theGT = theGT:cuda();

                        theSamples = theSamples:cuda();

                        theDist = theSamples - theGT;
                        theDist:cmul(theDist);
                        theDist = theDist:reshape(opt.numSamp, opt.batchsize, (36*opt.numSteps))
                        theDist = theDist:sum(3);

                        datafile = hdf5.open(opt.outDir .. "diff" .. tostring(curExamp) .. ".mat", "w");
                        datafile:write('/diff', theDist:double());
                        datafile:close();
                        return
                end





		theResult[1] = theResult[1]:reshape(opt.numSamp, opt.batchsize, 36*opt.numSteps);
		local U = torch.norm(theResult[3], 2, 3);
		U = U:cmul(torch.norm(theResult[1], 2,3));
		U = U:reshape(opt.numSamp, opt.batchsize);
		U = U:transpose(2,1);
		local trueSamps = {};

		trueSamps[1] = theResult[1]:clone();
		trueSamps[2] = theResult[2]:clone();
		trueSamps[3] = theResult[3]:clone();

		local portion = 0.1;

		for j = 1, opt.batchsize do
			temp,indexes=torch.sort(-1.0*U[j]);
			for i = 1, opt.numSamp*portion do
				trueSamps[1][i][j] = theResult[1][indexes[i]][j];
				trueSamps[2][i][j] = theResult[2][indexes[i]][j];
				trueSamps[3][i][j] = theResult[3][indexes[i]][j];
			end
		end

		theResult[2] = trueSamps[2]:narrow(1,1, opt.numSamp*portion):clone();
		theResult[1] = trueSamps[1]:narrow(1,1, opt.numSamp*portion):clone();
		theResult[3] = trueSamps[3]:narrow(1,1, opt.numSamp*portion):clone();

		theResult[1] = theResult[1]:reshape(opt.numSamp*portion, opt.batchsize, 36, opt.numSteps);

		local batch = opt.batchsize;

		theResult[1] = theResult[1]:reshape(opt.batchsize*opt.numSamp*portion, 36, opt.numSteps);
		theResult[2] = theResult[2]:reshape(opt.batchsize*opt.numSamp*portion, opt.numSteps, 54);
		theResult[3] = theResult[3]:reshape(opt.batchsize*opt.numSamp*portion, opt.numSteps*2);

		opt.batchsize = opt.batchsize*opt.numSamp*portion;

		local theSamples = getIntermediates(theResult[1], theResult[2]:clone(), theResult[3]):clone()

		opt.batchsize = batch;

		local numSamp = opt.numSamp;
		opt.numSamp = numSamp * portion;

		theSamples = theSamples:reshape(opt.numSamp, opt.batchsize, opt.timestep, 3, 18);
		local theSplitSamples = theSamples:narrow(4,1,2);
		local theConfs = theSamples:narrow(4,3,1);
		local theSplitSamples = theSplitSamples:reshape(opt.numSamp, opt.batchsize, opt.timestep*36);

		local sampMeans = theSplitSamples:mean(1);
		sampMeans = torch.repeatTensor(sampMeans, opt.numSamp, 1,1);
		local sampStd = theSplitSamples:std(1);
		sampStd = sampStd:reshape(1, opt.batchsize, opt.timestep*36);
		sampStd = torch.repeatTensor(sampStd, opt.numSamp, 1,1);

		theSplitSamples:add(-1.0*sampMeans);
		theSplitSamples:cdiv(sampStd+0.000000001);

		theClusters ={};
		theIndexes = {};
		for i = 1,opt.batchsize do
				
			theClusters[i],theIndexes[i] = unsup.kmeans(theSplitSamples:narrow(2,i,1):reshape(opt.numSamp,opt.timestep*36),
				opt.numClusts);

			theClusters[i] = theClusters[i]:reshape(1, opt.numClusts,opt.timestep*36) 


			theClusters[i]:cmul(torch.repeatTensor(sampStd[1][i]:reshape(1,1,opt.timestep*36), 
				1, opt.numClusts, 1)+0.000000001);

			theClusters[i]:add(torch.repeatTensor(sampMeans[1][i]:reshape(1,1,opt.timestep*36),
				1, opt.numClusts, 1));

			theClusters[i] = theClusters[i]:reshape(opt.numClusts, opt.timestep, 2, 18);

			local curConfs = theConfs[1][i];

			curConfs = curConfs:reshape(1,opt.timestep,1,18);

			curConfs = torch.repeatTensor(curConfs, opt.numClusts, 1, 1, 1);

			theClusters[i] = torch.cat(theClusters[i], curConfs, 3);

			theClusters[i] = theClusters[i]:reshape(1, opt.numClusts,opt.timestep*54) 
			

			local theMax = 0
			local aClust = 1;

			for j = 1, opt.numClusts do
				if(theIndexes[i][j] > theMax) then
					theMax = theIndexes[i][j];
					aClust = j;
				end
			end
				
			theClusters[i][1][1] = theClusters[i][1][aClust];
			theIndexes[i][1] = theIndexes[i][aClust];

			end

			joinTable = nn.JoinTable(1);
			joinTable = joinTable:cuda();

			sys.execute('rm '.. opt.outDir .. '*.mp4')

			theClusters = joinTable:forward(theClusters);

                        for j = 1, opt.printnumClusts do
			  bigClust = theClusters:narrow(2,j,1);
			  bigClust = bigClust:reshape(opt.batchsize, opt.timestep, 3, 18);
			  curClust = j;
			  makeIntermediate(bigClust, theInputs:narrow(2,1,3), 5,  theIndexes, curClust)
                        end

			opt.numSamp = numSamp;

	end
