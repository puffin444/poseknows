local LMDBTools = {}


function LMDBTools.loadH(theFile)
	--adapted from http://blog.aicry.com/torch7-reading-csv-into-tensor/
	local csvFile = io.open(theFile, 'r')  

	if(csvFile == nil) then
		local theMat = torch.Tensor(9)
		theMat = theMat:reshape(3,3);
		theMat = theMat:zero();
		theMat[1][1] = 1.0;
		theMat[2][2] = 1.0;
		theMat[3][3] = 1.0;
		return theMat
	end

	local header = csvFile:read()

	local theMat = torch.Tensor(9)
	
	local l = header:split(' ')
        local i = 1;
	  for key, val in ipairs(l) do
	    theMat[i] = val
            i = i + 1;
	  end


	csvFile:close()

	return theMat:reshape(3,3);
end


function LMDBTools.toLMDB(self, theDir, txn, count)
	local isFile = io.open(theDir.."/smoothPoses.mat", "r");

	if(isFile == nil) then
		return count
	end

	isFile.close();

	local datafile = hdf5.open(theDir.."/smoothPoses.mat", "r");
	local thePoseList = datafile:read('/thePoses'):all();

	if(thePoseList:max() < 1) then
		return count
	end

	thePoseList = thePoseList:transpose(3,1);
        local numPos = thePoseList:size()[1];
	datafile:close();

	for kk = 1,numPos do

		local thePoses = thePoseList[kk];

		local poseConf = ((thePoses:reshape(thePoses:size()[1],3,18)):transpose(2,1))[3];


		if (poseConf:max() > -1000000 and poseConf[poseConf:ge(0)]:mean() > opt.confThresh) then

                	print(poseConf[poseConf:ge(0)]:mean());

			local numJpgs = 1
			local jpgFiles = {}

			for file in paths.files(theDir) do
		  		if file:find(".jpg" .. '$') then
					jpgFiles[numJpgs] = file
					numJpgs = numJpgs + 1;
		  		end
			end

			table.sort(jpgFiles);
			numJpgs = numJpgs - 1;

			theChunk = opt.timestep/opt.numSteps;

			for i = 1,(thePoses:size()[1]-(opt.timestep+opt.numDecSteps*theChunk + 1)),15 do

				local theDiffs = torch.Tensor(opt.timestep+opt.numDecSteps*theChunk,18*3);
				theDiffs = theDiffs:float();
				theDiffs = theDiffs:zero();

				local contPoses = torch.Tensor(opt.timestep+opt.numDecSteps*theChunk,18*3);
				contPoses = contPoses:float();
				contPoses = contPoses:zero();

				local chunkPoses = torch.Tensor(opt.numSteps+opt.numDecSteps,18*3);
				chunkPoses = chunkPoses:float();
				chunkPoses = chunkPoses:zero();

				local theCompDiffs = torch.Tensor(36,opt.numSteps+opt.numDecSteps);
				theCompDiffs = theCompDiffs:zero();

				local firstPose = thePoses[i + 1];
				firstPose = firstPose:reshape(3,18);
				firstPose[1] = firstPose[1]/240.0;
				firstPose[2] = firstPose[2]/320.0;

				chunkPoses[1] = firstPose:clone();

				local curH = torch.Tensor(3,3);
				curH = curH:zero();
				curH[1][1] = 1.0;
				curH[2][2] = 1.0;
				curH[3][3] = 1.0;

				local prevPose = firstPose:clone();
		
				for j = (i + 2), (i+opt.timestep+opt.numDecSteps*theChunk + 1) do
					print({i, j, jpgFiles[j]})

					local nextH = self.loadH(theDir..string.sub(jpgFiles[j], 1, -5)..".txt");
					nextH = torch.inverse(nextH);
					curH = torch.mm(nextH, curH);

					local onePose = thePoses[j];
					onePose = onePose:reshape(3,18);
					onePose[1] = onePose[1]/240.0;
					onePose[2] = onePose[2]/320.0;

					prevPose = thePoses[torch.floor((j - i - 1)/theChunk - 0.001)*theChunk + i + 1]:clone();
					prevPose = prevPose:reshape(3,18);
					prevPose[1] = prevPose[1]/240.0;
					prevPose[2] = prevPose[2]/320.0;

					local bigPose = onePose:narrow(1,1,2);
					bigPose[1] = bigPose[1]*240.0;
					bigPose[2] = bigPose[2]*320.0;
					local rectPose = torch.cat(bigPose:double(), torch.zeros(1,18)+1.0,1);
					rectPose = torch.mm(curH:float(), rectPose:float()):clone();
					rectPose[1] = torch.cdiv(rectPose[1], rectPose[3]);
					rectPose[2] = torch.cdiv(rectPose[2], rectPose[3]);
					onePose[1] = rectPose[1]:clone()/240.0;
					onePose[2] = rectPose[2]:clone()/320.0;

					bigPose = prevPose:narrow(1,1,2);
					bigPose[1] = bigPose[1]*240.0;
					bigPose[2] = bigPose[2]*320.0;
					rectPose = torch.cat(bigPose:double(), torch.zeros(1,18)+1.0,1);
					rectPose = torch.mm(curH:float(), rectPose:float()):clone();
					rectPose[1] = torch.cdiv(rectPose[1], rectPose[3]);
					rectPose[2] = torch.cdiv(rectPose[2], rectPose[3]);
					prevPose[1] = rectPose[1]:clone()/240.0;
					prevPose[2] = rectPose[2]:clone()/320.0;


					theDiffs[j - i - 1] = (onePose:float()):reshape(18*3):float() - prevPose:reshape(18*3):float();
					contPoses[j - i - 1] = prevPose:clone();
				end

				theDiffs[theDiffs:lt(-1000)] = 0;
				theDiffs = theDiffs:transpose(2,1);

				for k = 1,(opt.numSteps+opt.numDecSteps) do
					chunkPoses[k] = contPoses[k*(opt.timestep/opt.numSteps)]:clone();
				end


				for j = 1, 36 do
					for k = 1,(opt.numSteps+opt.numDecSteps) do
						 local temp = theDiffs[j][k*(opt.timestep/opt.numSteps)];
						 theCompDiffs[j][k] = temp;
					end
				end


				local im = image.load(theDir .. jpgFiles[i+opt.numDecSteps*theChunk]);
				im = image.scale(im, 320,240);
				im = im*255.0;

				local paintPose = image.load(theDir .. '/../lmdb/pose_' ..
					 string.format('%06i', i+opt.numDecSteps*theChunk - 1)..".png");
				paintPose = image.scale(paintPose, 160,120);
				paintPose = paintPose*255.0;

				chunkPoses = chunkPoses:reshape(opt.numSteps+opt.numDecSteps,3,18);

				txn:put(count, theCompDiffs:float());
				txn:put(count+1, im:byte());
				txn:put(count+2, paintPose:byte());
				txn:put(count+3, chunkPoses:float());
				count = count + 4
			end
		end
	end

	return count;
end

return LMDBTools;
