local function doWarp(curH, height, width)
			local temp1 = imwarp(curH,height, width, 3);
			return temp1;
end

function loadup(theImages, theImagesPos, theHom)
	for k = 1,opt.batchsizeGAN do
		local h = ((math.random(opt.numDBGAN )) + 1)

		h = goodCounts[h][1];
		print(h)

		loadMDB = {};

		local curH = torch.Tensor(3,3);
		curH = curH:zero();
		curH[1][1] = 1.0;
		curH[2][2] = 1.0;
		curH[3][3] = 1.0;

		local temp3 = curH:clone();
		local temp4 = curH:clone();
		temp3[1][1] = 1.0/4.0;
		temp3[2][2] = 1.0/3.75;

                temp4[1][1] = 4.0;
                temp4[2][2] = 3.75;

		temp3 = temp3:float();
		temp4 = temp4:float();

		local mirrorH = curH:clone();

		if(math.random(2) > 1) then
			mirrorH[1][1] = -1.0;
			mirrorH[1][3] = 320;
		end

		local cropH = curH:clone();

		if(math.random(2) > 1) then
			cropH[1][3] = -math.random(64);
			cropH[2][3] = -math.random(48);

			cropH[1][1] = 1.25;
			cropH[2][2] = 1.25;
		end

		local transH = torch.mm(cropH:float(), mirrorH:float());

		for i = 1,(32) do
			loadMDB[i] = {}; 
			loadMDB[i][1] = (txn:get((i)*3+h)):float();

			loadMDB[i][2] = (txn:get((i)*3+h+1)):float();

			loadMDB[i][3] = torch.inverse(txn:get((i)*3+h+2)):float();
		end

		curH = curH:zero();
		curH[1][1] = 1.0;
		curH[2][2] = 1.0;
		curH[3][3] = 1.0;

		for i = 1,(32) do
			local tempH = torch.mm(transH, curH:float());
			temp5 = doWarp(torch.mm(temp3,torch.mm(torch.inverse(tempH:float()),temp4)), opt.height, opt.width);
			loadMDB[i][4] = temp5:clone();
			curH = torch.mm(loadMDB[i][3], curH:float());

		end

		for i = 1,(32) do
			theImages[k][i] = loadMDB[i][2]:clone()/255.0; 
			theImagesPos[k][i] = loadMDB[i][1]:clone()/255.0; 
			theHom[k][i] = loadMDB[i][4]:clone(); 
		end
	end
end
