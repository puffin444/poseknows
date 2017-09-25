function doScaling(X, Y, Z, theSteps)

	Y = Y:transpose(3,2);
	
	for i = 1,Y:size()[1] do
			for k = 1,theSteps do	
				local curVec = Y[i][k]:narrow(1,1,18);
				Z[i][(k-1)*2+1] = curVec:norm();
				local KJ = Y[i][k]:sub(1,18);
				KJ:div(Z[i][(k-1)*2+1] + 0.000000001)

				local curVec = Y[i][k]:narrow(1,19,18);
				Z[i][(k-1)*2+2] = curVec:norm();
				local KJ = Y[i][k]:sub(19,36);
				KJ:div(Z[i][(k-1)*2+2] + 0.000000001)

				Z[i][(k-1)*2+1] = (math.exp(Z[i][(k-1)*2+1])-opt.diffMean)/opt.diffStd;
				Z[i][(k-1)*2+2] = (math.exp(Z[i][(k-1)*2+2])-opt.diffMean)/opt.diffStd;
			end
	end

	Y = Y:transpose(3,2);

	Y = Y:transpose(2,1);

	for i = 1,36 do
		Y[i] = Y[i]*5.0;
	end

	Y = Y:transpose(2,1);

end

function doScalingInv(X, Y, Z, theSteps)

	Y = Y:transpose(2,1);

	for i = 1,36 do
		Y[i] = Y[i]/5.0;
	end

	Y = Y:transpose(2,1);

	Y = Y:transpose(3,2);

	for i = 1,Y:size()[1] do
			for k = 1,theSteps do	
				if(Z[i][(k-1)*2+1] < -0.2/opt.diffStd) then
					Z[i][(k-1)*2+1] = -0.2/opt.diffStd
				end

				if(Z[i][(k-1)*2+2] < -0.2/opt.diffStd) then
					Z[i][(k-1)*2+2] = -0.2/opt.diffStd
				end

				for l = 1,18 do
					Y[i][k][l] = Y[i][k][l]*math.log(Z[i][(k-1)*2+1]*opt.diffStd+opt.diffMean)
				end

				for l = 19,36 do
					Y[i][k][l] = Y[i][k][l]*math.log(Z[i][(k-1)*2+2]*opt.diffStd+opt.diffMean)
				end
			end
	
	end

	Y = Y:transpose(3,2);
end


function fulltransform(untransformed, transformations)
		untransformed = untransformed:reshape(opt.batchsize,6,320,240);
		transformations = transformations:reshape(opt.batchsize,2,320,240);

		untransformed = untransformed:transpose(4,2);
		transformations = transformations:transpose(4,2);

		transformations = transformations:clone();
		untransformed = untransformed:clone();

		local transformed = warper:forward({untransformed, transformations}):clone();
		transformed = transformed:transpose(4,2);

		transformed[transformed:ne(transformed)] = 0;

		return transformed
end

local function doWarp(curH, height, width)
			local temp1 = imwarp(curH,height, width, 3);
			return temp1;
end

function loadup(theImages, theHom, thePos, theChange,theRealPos, thePosDec, theChangeDec, 
			theMag, theMagDec)

	for kk = 1,opt.batchsize do
		local h = ((math.random(opt.numDB )) + 1 + opt.DBoffset)
		h = h*4;

		if(opt.testGen) then
			h = curBatch*4;
			curBatch = curBatch + 10;
		end

		local loadMDB = {};

		local curH = torch.Tensor(3,3);
		curH = curH:zero();
		curH[1][1] = 1.0;
		curH[2][2] = 1.0;
		curH[3][3] = 1.0;

		local temp3 = curH:clone();
		local temp4 = curH:clone();
		temp3[1][1] = 1.0/4.0;
		temp3[2][2] = 1.0/4.0;

		temp4[1][1] = 4.0;
		temp4[2][2] = 4.0;
		temp3 = temp3:float();
		temp4 = temp4:float();

		local mirrorH = curH:clone();

		if(math.random(2) > 1 and not opt.testGen) then
			mirrorH[1][1] = -1.0;
			mirrorH[1][3] = 320;
		end

		local cropH = curH:clone();

		if(math.random(2) > 1 and not opt.testGen) then
			cropH[1][3] = -math.random(64);
			cropH[2][3] = -math.random(48);

			cropH[1][1] = 1.25;
			cropH[2][2] = 1.25;
		end

		local transH = torch.mm(cropH:float(), mirrorH:float());


		loadMDB = {}; 
		loadMDB[1] = (txn:get(h)):float();

		local tempH = transH:clone()

		loadMDB[2] = (txn:get(h+1)):float();
		loadMDB[3] = (txn:get(h+2)):float();
		loadMDB[4] = (txn:get(h+3)):float();

		loadMDB[1] = loadMDB[1]:transpose(2,1);

		for k = 1,opt.numSteps+opt.numDecSteps do

			local bigPose = loadMDB[4][k]:narrow(1,1,2):clone();
			bigPose[1] = bigPose[1]*240.0;
			bigPose[2] = bigPose[2]*320.0;

			local rectPose = torch.cat(bigPose:double(), torch.zeros(1,18)+1.0,1);
			rectPose = torch.mm(tempH:float(), rectPose:float()):clone();
			rectPose[1] = torch.cdiv(rectPose[1], rectPose[3]);
			rectPose[2] = torch.cdiv(rectPose[2], rectPose[3]);
			rectPose[1] = rectPose[1]/240.0;
			rectPose[2] = rectPose[2]/320.0;

			local rectifiedTemp = rectPose:clone();
			rectifiedTemp[3] = loadMDB[4][k]:narrow(1,3,1):clone();
			rectifiedTemp[3][rectifiedTemp[3]:le(0.1)] = 0;
			rectifiedTemp[3][rectifiedTemp[3]:ge(0.1)] = 1.0;

		 	local tempCoord = loadMDB[1][k]:reshape(2,18);
			tempCoord = tempCoord + loadMDB[4][k]:narrow(1,1,2);
			bigPose = tempCoord:narrow(1,1,2);
			bigPose[1] = bigPose[1]*240.0;
			bigPose[2] = bigPose[2]*320.0;
			rectPose = torch.cat(bigPose:double(), torch.zeros(1,18)+1.0,1);
			rectPose = torch.mm(tempH:float(), rectPose:float()):clone();
			rectPose[1] = torch.cdiv(rectPose[1], rectPose[3]);
			rectPose[2] = torch.cdiv(rectPose[2], rectPose[3]);
			tempCoord[1] = rectPose[1]/240.0;
			tempCoord[2] = rectPose[2]/320.0;
			tempCoord[1] = (tempCoord[1] - rectifiedTemp[1])*1.0;
			tempCoord[2] = (tempCoord[2] - rectifiedTemp[2])*1.0;

			local temp2 = loadMDB[1][k]:reshape(2,18);
			temp2[1] = tempCoord[1]:clone();
			temp2[2] = tempCoord[2]:clone();
			loadMDB[1][k] = temp2:reshape(2*18):clone();
				
			if(k > opt.numDecSteps) then
				thePos[kk][k-opt.numDecSteps] = rectifiedTemp:clone();
			else
				thePosDec[kk][k] = rectifiedTemp:clone();
			end
		end

		loadMDB[1] = loadMDB[1]:transpose(2,1);

		curH = curH:zero();
		curH[1][1] = 1.0;
		curH[2][2] = 1.0;
		curH[3][3] = 1.0;

		local tempH = torch.mm(transH, curH:float());

		loadMDB[3] = (txn:get(h+2)):byte();
		loadMDB[4] = (txn:get(h+3)):float();

		local temp5 = doWarp(torch.mm(temp3,torch.mm(torch.inverse(tempH:float()),temp4)), 60, 80);
		loadMDB[5] = temp5:clone();

		theChange[kk] = loadMDB[1]:narrow(2,opt.numDecSteps+1, opt.numSteps):clone()
		theChangeDec[kk] = loadMDB[1]:narrow(2,1, opt.numDecSteps):clone()

		theImages[kk] = loadMDB[2]:clone(); 
		theRealPos[kk] = (loadMDB[3]:clone()); 
		theHom[kk] = loadMDB[5]:clone(); 
	end

	if(opt.testGen) then
	       if(not opt.numEval) then
                 theChange = theChange:zero();
               end
       	       thePos:narrow(2,2,4):zero();
	end

	theImages = theImages/255.0;
	theRealPos = theRealPos/255.0;

	thePos[thePos:ge(1000)] = 0;
	theChange[theChange:ge(1000)] = 0;

	thePos[thePos:le(-1000)] = 0;
	theChange[theChange:le(-1000)] = 0;

	thePos[thePos:ne(thePos)] = 0;
	theChange[theChange:ne(theChange)] = 0;

	thePosDec[thePosDec:ge(1000)] = 0;
	theChangeDec[theChangeDec:ge(1000)] = 0;

	thePosDec[thePosDec:le(-1000)] = 0;
	theChangeDec[theChangeDec:le(-1000)] = 0;

	thePosDec[thePosDec:ne(thePosDec)] = 0;
	theChangeDec[theChangeDec:ne(theChangeDec)] = 0;

	theMag:zero();
	theMagDec:zero();

	doScaling(thePos, theChange, theMag, opt.numSteps);
	doScaling(thePosDec, theChangeDec, theMagDec, opt.numDecSteps);

	thePos[thePos:ge(10)] = 0;
	theChange[theChange:ge(10)] = 0;
	theMag[theMag:ge(10)] = 0;

	thePos[thePos:le(-10)] = 0;
	theChange[theChange:le(-10)] = 0;

	thePos[thePos:ne(thePos)] = 0;
	theChange[theChange:ne(theChange)] = 0;
	theMag[theMag:ne(theMag)] = 0;

	thePosDec[thePosDec:ge(10)] = 0;
	theChangeDec[theChangeDec:ge(10)] = 0;
	theMagDec[theMagDec:ge(10)] = 0;

	thePosDec[thePosDec:le(-10)] = 0;
	theChangeDec[theChangeDec:le(-10)] = 0;

	thePosDec[thePosDec:ne(thePosDec)] = 0;
	theChangeDec[theChangeDec:ne(theChangeDec)] = 0;
	theMagDec[theMagDec:ne(theMagDec)] = 0;
end
