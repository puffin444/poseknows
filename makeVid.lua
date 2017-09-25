function makeVid(tempChange, tempPos, tempMag, theImages, theSignal, theSteps)

	doScalingInv(tempPos, tempChange,tempMag, theSteps)

	local theDiffs = torch.Tensor(opt.timestep,18*3);
	theDiffs = theDiffs:cuda();
	theDiffs = theDiffs:transpose(2,1);
	theImages = theImages:reshape(opt.batchsize, 3, 240, 320);

	local numSeq = tempMag:size()[2];

	local theChunk = opt.timestep/theSteps;

        for m = 1,opt.printLimit do
		theDiffs = theDiffs:zero();
			for j = 1, 36 do
				for hh = 1,theSteps do
					local localDiff = torch.Tensor(theChunk,18*3);
					localDiff = localDiff:cuda();
					localDiff = localDiff:transpose(2,1);

				        for k = 1,theChunk do
					    theDiffs[j][(hh - 1)*theChunk + k] = theDiffs[j][(hh - 1)*theChunk + k] 
						+ tempChange[m][j][hh]/(opt.timestep/theSteps)*1.0*k
				        end


				end
			end
		


			theDiffs = theDiffs:transpose(2,1);

			theChunk = opt.timestep/theSteps;

			for hh = 1,theSteps do
				for k = 1,theChunk do
					theDiffs[(hh - 1)*theChunk + k] = theDiffs[(hh - 1)*theChunk + k] + tempPos[m][hh]
				end
			end

			local alpha = 0.25;
			local st = theDiffs[1];

			for hh = 1,(opt.timestep - 1) do
				theDiffs[hh+1] = alpha*theDiffs[hh+1] + (1-alpha)*st:clone();
				st = theDiffs[hh+1]:clone();
			end
	
			local curImage = theImages[m]:clone();

			if(theSignal == 3 or theSignal == 4 or theSignal == 5) then
				curImage = theImages[m]:clone();
			end

			startPose = tempPos[m][1]:reshape(3,18);

			paintPose((curImage), startPose, 1)
	
			for j = 1,(opt.timestep) do
				local temp = theDiffs[j]:reshape(3,18);
				paintPose((curImage), temp, j+1)
			end

			theDiffs = theDiffs:transpose(2,1);
		

			if(theSignal == 1) then
				sys.execute('rm ' .. opt.outDir .. 'outGT_' .. tostring(m) .. '_.mp4')
				sys.execute('/nfs/hn48/jcwalker/ffmpeg -framerate 15 -i ' .. opt.outDir .. '%06d.png -c:v libx264 -r 30 -pix_fmt yuv420p ' .. opt.outDir .. 'outGT_' .. tostring(m) .. '_.mp4')
				sys.execute('rm '.. opt.outDir .. '*.png')
			end


			if(theSignal == 2) then
				sys.execute('rm ' .. opt.outDir .. 'outPredA_' .. tostring(m) .. '_.mp4')
				sys.execute('/nfs/hn48/jcwalker/ffmpeg -framerate 15 -i ' .. opt.outDir .. '%06d.png -c:v libx264 -r 30 -pix_fmt yuv420p ' .. opt.outDir .. 'outPredA_' .. tostring(m) .. '_.mp4')
				sys.execute('rm '.. opt.outDir .. '*.png')
			end

			if(theSignal == 3) then
				sys.execute('rm ' .. opt.outDir .. 'outPred_' .. tostring(m) .. '_.mp4')
				sys.execute('/nfs/hn48/jcwalker/ffmpeg -framerate 15 -i ' .. opt.outDir .. '%06d.png -c:v libx264 -r 30 -pix_fmt yuv420p ' .. opt.outDir .. 'outPred_' .. tostring(m) .. '_.mp4')
				sys.execute('rm '.. opt.outDir .. '*.png')
			end

			if(theSignal == 4) then
				sys.execute('rm ' .. opt.outDir .. 'outPredSamp_' .. tostring(m) .. '_.mp4')
				sys.execute('/nfs/hn48/jcwalker/ffmpeg -framerate 15 -i ' .. opt.outDir .. '%06d.png -c:v libx264 -r 30 -pix_fmt yuv420p ' .. opt.outDir .. 'outPredSamp_' .. tostring(m) .. '_.mp4')
				sys.execute('rm '.. opt.outDir .. '*.png')
			end

			if(theSignal == 5) then
				sys.execute('rm ' .. opt.outDir .. 'outPredClust_' .. tostring(L[m][curClust]) .. '_' .. tostring(m) .. '_.mp4')
				sys.execute('/nfs/hn48/jcwalker/ffmpeg -framerate 15 -i ' .. opt.outDir .. '%06d.png -c:v libx264 -r 30 -pix_fmt yuv420p ' .. opt.outDir .. 'outPredClust_' .. tostring(L[m][curClust]) .. '_' .. tostring(m) .. '_.mp4')
				sys.execute('rm '.. opt.outDir .. '*.png')
			end

			if(theSignal == 6) then
				sys.execute('rm ' .. opt.outDir .. 'outDec_' .. tostring(m) .. '_.mp4')
				sys.execute('/nfs/hn48/jcwalker/ffmpeg -framerate 15 -i ' .. opt.outDir .. '%06d.png -c:v libx264 -r 30 -pix_fmt yuv420p ' .. opt.outDir .. 'outDec_' .. tostring(m) .. '_.mp4')
				sys.execute('rm '.. opt.outDir .. '*.png')
			end

			if(theSignal == 7) then
				sys.execute('rm ' .. opt.outDir .. 'outDecGT_' .. tostring(m) .. '_.mp4')
				sys.execute('/nfs/hn48/jcwalker/ffmpeg -framerate 15 -i ' .. opt.outDir .. '%06d.png -c:v libx264 -r 30 -pix_fmt yuv420p ' .. opt.outDir .. 'outDecGT_' .. tostring(m) .. '_.mp4')
				sys.execute('rm '.. opt.outDir .. '*.png')
			end

	end

	theImages = theImages:reshape(opt.batchsize, 3, 240, 320);
	doScaling(tempPos, tempChange,tempMag,theSteps)

end
