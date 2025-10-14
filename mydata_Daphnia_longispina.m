function [data, auxData, metaData, txtData, weights] = mydata_Daphnia_longispina

%% set metaData
metaData.phylum     = 'Arthropoda'; 
metaData.class      = 'Branchiopoda'; 
metaData.order      = 'Anomopoda'; 
metaData.family     = 'Daphniidae';
metaData.species    = 'Daphnia_longispina'; 
metaData.species_en = 'Waterflea'; 

metaData.ecoCode.climate = {'Cfa', 'Cfb', 'Dfa', 'Dfb', 'Dfc'};
metaData.ecoCode.ecozone = {'TH'};
metaData.ecoCode.habitat = {'0iFp', '0iFl', '0iFm'};
metaData.ecoCode.embryo  = {'Fbf'};
metaData.ecoCode.migrate = {};
metaData.ecoCode.food    = {'biPp', 'biD'};
metaData.ecoCode.gender  = {'D'};
metaData.ecoCode.reprod  = {'Apf'};

metaData.T_typical  = C2K(20); % K, body temp
metaData.data_0     = {'ab_f'; 'ap_f'; 'am_f'; 'Lb'; 'Lp'; 'Li'; 'Wdb'; 'Wdp'; 'Wdi'; 'Ni'}; 
metaData.data_1     = {'t-L_f'}; 

metaData.COMPLETE = 2.7; % using criteria of LikaKear2011

metaData.author   = {'Bas Kooijman'};    
metaData.date_subm = [2017 08 22];              
metaData.email    = {'bas.kooijman@vu.nl'};            
metaData.address  = {'VU University Amsterdam'};   

metaData.curator     = {'Starrlight Augustine'};
metaData.email_cur   = {'starrlight.augustine@akvaplan.niva.no'}; 
metaData.date_acc    = [2017 08 22]; 

%% set data
% zero-variate data

data.ab = 38/24;   units.ab = 'd';    label.ab = 'age at birth; well fed';             bibkey.ab = 'InglWood1937';   
  temp.ab = C2K(25);  units.temp.ab = 'K'; label.temp.ab = 'temperature';
  comment.ab = 'Deduced from instar duration after nr 6';
data.ab0 = 42/24;   units.ab0 = 'd';    label.ab0 = 'age at birth; starved';             bibkey.ab0 = 'InglWood1937';   
  temp.ab0 = C2K(25);  units.temp.ab0 = 'K'; label.temp.ab0 = 'temperature';
  comment.ab0 = 'Deduced from instar duration after nr 6';
data.tp = 148/24;   units.tp = 'd';    label.tp = 'time since birth at puberty, well fed'; bibkey.tp = 'InglWood1937';
  temp.tp = C2K(25);  units.temp.tp = 'K'; label.temp.tp = 'temperature';
data.tp0 = 174/24;   units.tp0 = 'd';    label.tp0 = 'time since birth at puberty,starved'; bibkey.tp0 = 'InglWood1937';
  temp.tp0 = C2K(25);  units.temp.tp0 = 'K'; label.temp.tp0 = 'temperature';
data.am = 28.2;    units.am = 'd';    label.am = 'life span; well fed';                bibkey.am = 'InglWood1937';   
  temp.am = C2K(25);  units.temp.am = 'K'; label.temp.am = 'temperature'; 
data.am0 = 38.5;    units.am0 = 'd';    label.am0 = 'life span; starved';                bibkey.am0 = 'InglWood1937';   
  temp.am0 = C2K(25);  units.temp.am0 = 'K'; label.temp.am0 = 'temperature'; 

data.Lb  = 0.0627; units.Lb  = 'cm';  label.Lb  = 'body length at birth';   bibkey.Lb  = 'InglWood1937';
data.Lp  = 0.2132; units.Lp  = 'cm';  label.Lp  = 'body length at puberty, well-fed'; bibkey.Lp  = 'InglWood1937';
data.Lp0  = 0.1562; units.Lp0  = 'cm';  label.Lp0  = 'body length at puberty, starved'; bibkey.Lp0  = 'InglWood1937';
data.Li  = 0.290; units.Li  = 'cm';  label.Li  = 'ultimate body length, well fed';   bibkey.Li  = 'InglWood1937';
data.Li0  = 0.210; units.Li0  = 'cm';  label.Li0  = 'ultimate body length, starved';   bibkey.Li0  = 'InglWood1937';

data.Wdb = 3.33e-6; units.Wdb = 'g'; label.Wdb = 'dry weight at birth';     bibkey.Wdb = 'InglWood1937';
  comment.Wdb = 'Computed as 1.16e-5*(Lb * 10)^2.67, see F1';
data.Wdp = 8.76e-5; units.Wdp = 'g'; label.Wdp = 'dry weight at puberty';   bibkey.Wdp = 'InglWood1937';
  comment.Wdp = 'Computed as 1.16e-5*(Lp * 10)^2.67, see F1';
data.Wdi = 1.99e-4; units.Wdi = 'g'; label.Wdi = 'ultimate dry weight';     bibkey.Wdi = 'InglWood1937';
  comment.Wdi = 'Computed as 1.16e-5*(Li * 10)^2.67, see F1';

data.Ni  = 263;   units.Ni  = '#'; label.Ni  = 'life time reprod output; well fed';    bibkey.Ni  = 'InglWood1937';   
  temp.Ni = C2K(25);  units.temp.Ni = 'K'; label.temp.Ni = 'temperature';
data.Ni0  = 88;   units.Ni0  = '#'; label.Ni0  = 'life time reprod output; starved';  bibkey.Ni0  = 'InglWood1937';   
  temp.Ni0 = C2K(25);  units.temp.Ni0 = 'K'; label.temp.Ni0 = 'temperature';
 
% uni-variate data
% time-length
% well-fed
tL = [ ... % instar (d), body length (mm), duration (h)
     1 0.627 16.59
     2 0.810 16.02
     3 1.126 19.07
     4 1.487 23.41
     5 1.894 35.58
     6 2.132 37.54
     7 2.454 38.14
     8 2.454 38.52
     9 2.550 39.45
    10 2.681 39.00
    11 2.732 46.67
    12 2.775 42.94
    13 2.822 47.46
    14 2.866 45.10
    15 2.938 51.53
    16 2.956 55.71
    17 2.932 57.96
    18 2.922 54.71
    19 2.916 58.71
    20 2.884 55.50
    21 2.858 55.00 % instar duration guessed
    22 2.750 55.00 % instar duration guessed
    ];
data.tL = [[0;cumsum(tL(1:21,3))]/ 24, tL(:,2)/ 10]; % compose tL 
units.tL   = {'d', 'cm'};  label.tL = {'time since birth', 'body length', 'well-fed'};  
temp.tL    = C2K(25);  units.temp.tL = 'K'; label.temp.tL = 'temperature';
bibkey.tL = 'InglWood1937';
comment.tL = 'Well-fed; young at instar 6';
% starved
tL0 = [ ... % instar (d), body length (mm), duration (h)
     1 0.626 17.25
     2 0.794 17.96
     3 1.032 22.96
     4 1.319 27.98
     5 1.508 43.08
     6 1.562 44.67
     7 1.637 42.31
     8 1.756 42.73
     9 1.826 45.28
    10 1.853 46.59
    11 1.956 43.65
    12 1.968 47.39
    13 2.010 45.43
    14 2.065 48.55
    15 2.091 55.22
    16 2.109 50.27
    17 2.127 56.53
    18 2.140 58.07
    19 2.147 60.89
    20 2.156 68.07
    21 2.165 59.50
    22 2.169 60.00
    23 2.190 60.00 % instar duration guessed
    ];
data.tL0 = [[0;cumsum(tL0(1:22,3))]/ 24, tL0(:,2)/ 10]; % compose tL 
units.tL0   = {'d', 'cm'};  label.tL0 = {'time since birth', 'body length', 'starved'};  
temp.tL0    = C2K(25);  units.temp.tL0 = 'K'; label.temp.tL0 = 'temperature';
bibkey.tL0 = 'InglWood1937';
comment.tL0 = 'starved; young at instar 6';
 
%% set weights for all real data
weights = setweights(data, []);
weights.tL = 5 * weights.tL;
weights.tL0 = 5 * weights.tL0;

%% set pseudodata and respective weights
[data, units, label, weights] = addpseudodata(data, units, label, weights);
weights.psd.p_M = weights.psd.p_M * 5;

%% pack auxData and txtData for output
auxData.temp = temp;
txtData.units = units;
txtData.label = label;
txtData.bibkey = bibkey;
txtData.comment = comment;

%% Group plots
set1 = {'tL', 'tL0'}; subtitle1 = {'Data for well-fed, starved'};
metaData.grp.sets = {set1};
metaData.grp.subtitle = {subtitle1};

%% Facts
F1 = 'Weight - length: Dry weight in mg = 0.0116*(body length in mm)^2.67';
metaData.bibkey.F1 = 'Bums1969'; 
metaData.facts = struct('F1',F1);

%% Links
metaData.links.id_CoL = '6CCSS'; % Cat of Life
metaData.links.id_ITIS = '83875'; % ITIS
metaData.links.id_EoL = '46498266'; % Ency of Life
metaData.links.id_Wiki = 'Daphnia_(subgenus)'; % Wikipedia
metaData.links.id_ADW = 'Daphnia_longispina'; % ADW
metaData.links.id_Taxo = '33104'; % Taxonomicon
metaData.links.id_WoRMS = '148373'; % WoRMS

%% References
bibkey = 'Wiki'; type = 'Misc'; bib = ...
'howpublished = {\url{http://en.wikipedia.org/wiki/Daphnia_longispina}}';
metaData.biblist.(bibkey) = ['''@', type, '{', bibkey, ', ' bib, '}'';'];
%
bibkey = 'Kooy2010'; type = 'Book'; bib = [ ...  % used in setting of chemical parameters and pseudodata
'author = {Kooijman, S.A.L.M.}, ' ...
'year = {2010}, ' ...
'title  = {Dynamic Energy Budget theory for metabolic organisation}, ' ...
'publisher = {Cambridge Univ. Press, Cambridge}, ' ...
'pages = {Table 4.2 (page 150), 8.1 (page 300)}, ' ...
'howpublished = {\url{../../../bib/Kooy2010.html}}'];
metaData.biblist.(bibkey) = ['''@', type, '{', bibkey, ', ' bib, '}'';'];
%
bibkey = 'InglWood1937'; type = 'Article'; bib = [ ... 
'author = {L. Ingle and T.R. Wood and A. M. Banta}, ' ... 
'year = {1937}, ' ...
'title = {A STUDY OF LONGEVITY, GROWTH, REPRODUCTION AND HEART RATE IN \emph{Daphnia longispina} QUANTITY OF FOOD AS INFLUENCED}, ' ...
'journal = {J. Exp. Zool.}, ' ...
'number = {2}, ' ...
'volume = {76}, ' ...
'pages = {325--352}'];
metaData.biblist.(bibkey) = ['''@', type, '{', bibkey, ', ' bib, '}'';'];
%
bibkey = 'Bums1969'; type = 'Article'; bib = [ ... 
'author = {C. W. Bums}, ' ... 
'year = {1969}, ' ...
'title = {RELATION BETWEEN FILTERING RATE, TEMPERATURE, AND BODY SIZE IN FOUR SPECIES OF \emph{Daphnia}}, ' ...
'journal = {ASLO}, ' ...
'volume = {15}, ' ...
'pages = {693--700}'];
metaData.biblist.(bibkey) = ['''@', type, '{', bibkey, ', ' bib, '}'';'];

