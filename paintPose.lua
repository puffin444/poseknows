function paintPose(theJpg, pose, index)

pose[1] = pose[1]*240.0;
pose[2] = pose[2]*320.0;

local theImage = (theJpg);
local datag,stride = Cairo.tensor2rgb24(theImage);
local surface = Cairo.image_surface_create_from_data(datag, 'rgb24', theImage:size(3), theImage:size(2) ,stride);
local cr = Cairo.context_create(surface);

local connected = {{2, 3}, {2, 6}, {3, 4}, {4, 5}, {6, 7}, {7, 8}, {2, 9}, {9, 10}, {10, 11}, {2, 12}, {12, 13}, {13, 14}, {2, 1}, {1, 15}, {15, 17}, {1, 16},{16, 18}, {3, 17}, {6, 18}};

local colors = {{255, 0, 0},  {255, 85, 0},  {255, 170, 0},  {255, 255, 0},  {170, 255, 0},   {85, 255, 0},  {0, 255, 0},  {0, 255, 85},  {0, 255, 170},  {0, 255, 255},  {0, 170, 255},  {0, 85, 255},  {0, 0, 255},   {85, 0, 255},  {170, 0, 255},  {255, 0, 255},  {255, 0, 170},  {255, 0, 85}};


local numjoints = table.getn(connected);

for i = 1,18 do
	if(pose[3][connected[i][1]] > 0.1 and pose[3][connected[i][2]] > 0.1) then

		cr:move_to(pose[1][connected[i][1]],pose[2][connected[i][1]]);
		cr:line_to(pose[1][connected[i][2]],pose[2][connected[i][2]]);
		cr:set_source_rgb(colors[i][1],colors[i][2],colors[i][3]);
		cr:set_line_width(3);
		cr:stroke()

		cr:move_to(pose[1][connected[i][1]],pose[2][connected[i][1]]);
		cr:arc(pose[1][connected[i][1]],pose[2][connected[i][1]],1,0,4*math.asin(1))
		cr:set_source_rgb(colors[i][1],colors[i][2],colors[i][3]);
		cr:set_line_width(3);
		cr:stroke()

	end
   end

surface:write_to_png(opt.outDir .. string.format('%06i', index)..".png")

end


