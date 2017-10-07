function printVideo(inputImages, flag, curNumber)

	for i = 1,30 do
		image.save(opt.outDirGAN .. '/' .. string.format('%06i', i)..".png", inputImages[i])
	end


	if(flag == 1) then
	sys.execute('rm ' .. opt.outDirGAN .. 'outPred_' .. string.format('%06i', curNumber) .. '.mp4')
	sys.execute('/nfs/hn48/jcwalker/ffmpeg -framerate 15 -i ' .. opt.outDirGAN .. '%06d.png -c:v libx264 -r 30 -pix_fmt yuv420p ' .. opt.outDirGAN .. 'outPred_' .. string.format('%06i', curNumber) .. '.mp4')
	sys.execute('rm '.. opt.outDirGAN .. '*.png')
	end

	if(flag == 2) then
	sys.execute('rm ' .. opt.outDirGAN .. 'outPred_fake_' .. string.format('%06i', curNumber) .. '.mp4')
	sys.execute('/nfs/hn48/jcwalker/ffmpeg -framerate 15 -i ' .. opt.outDirGAN .. '%06d.png -c:v libx264 -r 30 -pix_fmt yuv420p ' .. opt.outDirGAN .. 'outPred_fake_' .. string.format('%06i', curNumber) .. '.mp4')
	sys.execute('rm '.. opt.outDirGAN .. '*.png')
	end


end
