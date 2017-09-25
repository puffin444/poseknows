require 'lmdb'
require 'image'
require 'hdf5'
require 'io'
LMDBTools = require 'LMDBTools'

dofile('opts.lua');

local db = lmdb.env{
	Path = opt.dbPath,
	Name = opt.dbName
}

db:open();
print(db:stat());

local txn = db:txn();
local cursor = txn:cursor();
local count = 0;
local theDir = {}
local dirFiles = {}
local j = 0;
	
for file in paths.files(opt.vidFolder) do
	print(file)
  	if not file:find(".avi" .. '$') then
		dirFiles[j] = file
		j = j + 1;
  	end
end

table.sort(dirFiles);
local numDirs = table.getn(dirFiles);

local goodExp = torch.Tensor(1,1);
goodExp:zero();

for i = 1,opt.numVids do
		print(i)
		data[i] = {};
		data[i].theJpgs = {};
		local oldcount = count;

		print(dirFiles[i]);
		local newcount = LMDBTools.toLMDB(LMDBTools, opt.vidFolder .. dirFiles[i] .. "/images/", 
					txn, count)

		print(count)
		if(newcount > oldcount) then	
			txn:commit();
			txn = db:txn();
			cursor = txn:cursor();
			count = newcount;
		end
		torch.save(opt.dbPath .. '/counts.t7', count)
end

txn:commit();
txn = db:txn();
cursor = txn:cursor();
torch.save(opt.dbPath .. '/counts.t7', count)


