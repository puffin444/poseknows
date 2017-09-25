dpt = torch.load(opt.inDir .. "dpt_" .. tostring(opt.loadIter) .. ".dat");
dpt = dpt:cuda();

vae = torch.load(opt.inDir .. "vae_" .. tostring(opt.loadIter) .. ".dat");
vae = vae:cuda();

lstm = torch.load(opt.inDir .. "lstm_" .. tostring(opt.loadIter) .. ".dat");
lstm = lstm:cuda();

lstm_decoder = torch.load(opt.inDir .. "dec_" .. tostring(opt.loadIter) .. ".dat");
lstm_decoder = lstm_decoder:cuda();


