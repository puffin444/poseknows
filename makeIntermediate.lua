function makeIntermediate(thePaths, theImages, theSignal, theIndexes, curClust)

        for m = 1,opt.printLimit do

		local curImage = theImages[m]:clone();

		for j = 1,(opt.timestep) do
			local temp = thePaths[m][j]:reshape(3,18);
			paintPose((curImage), temp, j)
		end

		if(theSignal == 1) then
			sys.execute('rm ' .. opt.outDir .. 'outGT_' .. tostring(m) .. '_.mp4')
			sys.execute('/nfs/hn48/jcwalker/ffmpeg -framerate 15 -i ' .. opt.outDir .. '%06d.png -c:v libx264 -r 30 -pix_fmt yuv420p ' .. opt.outDir .. 'outGT_' .. tostring(m) .. '_.mp4')
			sys.execute('rm '.. opt.outDir .. '*.png')
		end


		if(theSignal == 2) then
			sys.execute('rm ' .. opt.outDir .. 'outDec_' .. tostring(m) .. '_.mp4')
			sys.execute('/nfs/hn48/jcwalker/ffmpeg -framerate 15 -i ' .. opt.outDir .. '%06d.png -c:v libx264 -r 30 -pix_fmt yuv420p ' .. opt.outDir .. 'outDec_' .. tostring(m) .. '_.mp4')
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
			sys.execute('/nfs/hn48/jcwalker/ffmpeg -framerate 15 -i ' .. opt.outDir .. '%06d.png -c:v libx264 -r 30 -pix_fmt yuv420p -y ' .. opt.outDir .. 'outPredClust_' .. tostring(m) .. '_' ..  tostring(theIndexes[m][curClust]) .. '_' .. tostring(curBatch) .. '.mp4')
			sys.execute('rm '.. opt.outDir .. '*.png')
		end

			sys.execute('rm '.. opt.outDir .. '*.png')
	end

end
