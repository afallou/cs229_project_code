dataDir = '/Users/adrien/Dropbox/Stanford/Massive big data/data';
resultsDir = '/Users/adrien/Dropbox/Stanford/Massive big data/output';

inFile = fullfile(dataDir,'adrien_face_2.mp4');
amplify_spatial_Gdown_temporal_ideal(inFile, resultsDir, 50, 4, 30/60, 240/60,30, 1);