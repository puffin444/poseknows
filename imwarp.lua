function imwarp(H,width, height, nchan)
	---Adapted from https://github.com/torch/image/blob/master/test/test_warp.lua
	local grid_y = torch.ger( torch.linspace(0,1,height), torch.ones(width) )
	local grid_x = torch.ger( torch.ones(height), torch.linspace(0,1,width) )
	grid_y = grid_y*(height - 1)+1.0;
	grid_x = grid_x*(width - 1)+1.0;

	local flow = torch.FloatTensor()
	flow:resize(2,height,width)
	flow[1] = grid_x:clone();
	flow[2] = grid_y:clone();

	local thecoords = torch.FloatTensor(3,height*width);

	thecoords[1] = grid_x:reshape(height*width):clone();
	thecoords[2] = grid_y:reshape(height*width):clone();
	thecoords[3] = thecoords[3]:zero() + 1.0;

	local trans = torch.eye(3);
	trans[1][1] = 0;
	trans[1][2] = 1;
	trans[2][1] = 1;
	trans[2][2] = 0;

	local thefinalcoords = torch.mm(torch.mm(H, trans:float()), thecoords);
	thefinalcoords[1] = torch.cdiv(thefinalcoords[1],thefinalcoords[3]);
	thefinalcoords[2] = torch.cdiv(thefinalcoords[2],thefinalcoords[3]);

	flow_scale = torch.FloatTensor()
	flow_scale:resize(2,height,width)
	flow_scale[1] = thefinalcoords[1]:reshape(height,width);
	flow_scale[2] = thefinalcoords[2]:reshape(height,width);

	flow[1] = ((flow_scale[1] - 1.0)/(height - 1))*2.0 - 1.0;
	flow[2] = ((flow_scale[2] - 1.0)/(width - 1))*2.0 - 1.0;

	flow_scale[1] = flow[2];
	flow_scale[2] = flow[1];
	
	return flow_scale:reshape(1,2,height,width);
end

