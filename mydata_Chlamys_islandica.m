function [data, auxData, metaData, txtData, weights] = mydata_Chlamys_islandica
%% set metaData
metaData.phylum     = 'Mollusca'; 
metaData.class      = 'Bivalvia'; 
metaData.order      = 'Pectinida'; 
metaData.family     = 'Pectinidae';
metaData.species    = 'Chlamys_islandica'; 
metaData.species_en = 'Iceland scallop'; 

metaData.ecoCode.climate = {'MC'};
metaData.ecoCode.ecozone = {'MAN'};
metaData.ecoCode.habitat = {'0jMp', 'jiMb'};
metaData.ecoCode.embryo  = {'Mp'};
metaData.ecoCode.migrate = {};
metaData.ecoCode.food    = {'biPp'};
metaData.ecoCode.gender  = {'D'};
metaData.ecoCode.reprod  = {'O'};

metaData.T_typical  = C2K(2); % K, body temp
metaData.data_0     = {'ab'; 'aj'; 'am'; 'Lp'; 'Li'; 'Wwb'; 'WdR_L'}; 
metaData.data_1     = {'t-L_f';'L-Wd_f';}; 

metaData.COMPLETE = 2.7; % using criteria of LikaKear2011

metaData.author   = {'Bas Kooijman','Starrlight Augustine'};    
metaData.date_subm = [2017 02 12];              
metaData.email    = {'bas.kooijman@vu.nl'};            
metaData.address  = {'VU University Amsterdam'};   

metaData.curator     = {'Starrlight Augustine'};
metaData.email_cur   = {'sta@akvaplan.niva.no'}; 
metaData.date_acc    = [2017 02 12]; 

%% set data
% zero-variate data

data.ab = 8;      units.ab = 'd';    label.ab = 'age at birth';             bibkey.ab = 'guess';   
  temp.ab = C2K(2);  units.temp.ab = 'K'; label.temp.ab = 'temperature';
  comment.ab = 'Guess based on 2 d for Pecten maximus at 18 C';
data.tj = 75;    units.tj = 'd';    label.tj = 'time since birth at metam'; bibkey.tj = 'guess';
  temp.tj = C2K(2);  units.temp.tj = 'K'; label.temp.tj = 'temperature';
  comment.tj = 'Guess based on 25 d for Pecten maximus at 13 C';
data.am = 23*365;   units.am = 'd';    label.am = 'life span';                bibkey.am = 'sealifebase';   
  temp.am = C2K(2);  units.temp.am = 'K'; label.temp.am = 'temperature'; 

data.Lp  = 2.45;   units.Lp  = 'cm';  label.Lp  = 'shell height at puberty'; bibkey.Lp  = 'BlicRysg2010';
data.Li  = 11;   units.Li  = 'cm';  label.Li  = 'ultimate shell height';   bibkey.Li  = 'sealifebase';

data.Wwb = 1.8e-7; units.Wwb = 'g';   label.Wwb = 'wet weight at birth';   bibkey.Wwb = 'ShumPars2016';
  comment.Wwb = 'Based on 70 mum egg diameter: 4/3*pi*0.0035^3';
data.WdR = 4.5;    units.WdR = 'g';    label.WdR = 'gonad dry weight at shell height 6.7 cm'; bibkey.WdR = 'BlicRysg2010';
  temp.WdR = C2K(2);  units.temp.WdR = 'K'; label.temp.WdR = 'temperature';
   
% uni-variate data

% time-length
% grp 1, 15 m deep
data.tL115 = [ ... % time since 2007/04/01 (d), shell height (cm)
43.570	2.619
130.709	2.937
169.347	3.220
203.874	3.427
245.800	3.623
290.191	3.766
337.050	3.876
371.577	3.952
401.171	4.062
435.698	4.150
461.182	4.237
485.023	4.357
554.899	4.641];
units.tL115   = {'d', 'cm'};  label.tL115 = {'time', 'shell height', 'grp 1, 15 m'};  
temp.tL115    = C2K(3);  units.temp.tL115 = 'K'; label.temp.tL115 = 'temperature';
bibkey.tL115 = 'BlicRysg2010';
comment.tL115 = 'Data for grp 1, 15 m deep';
% grp 1, 30 m deep
data.tL130 = [ ... % time since 2007/04/01 (d), shell height (cm)
44.392	2.554
130.709	2.882
170.169	3.090
203.052	3.242
246.622	3.396
289.369	3.505
337.050	3.572
371.577	3.692
401.171	3.747
434.876	3.878
460.360	3.923
485.023	4.010
553.254	4.262];
units.tL130   = {'d', 'cm'};  label.tL130 = {'time', 'shell height', 'grp 1, 30 m'};  
temp.tL130    = C2K(2);  units.temp.tL130 = 'K'; label.temp.tL130 = 'temperature';
bibkey.tL130 = 'BlicRysg2010';
comment.tL130 = 'Data for grp 1, 30 m deep';
% grp 2, 15 m deep
data.tL215 = [ ... % time since 2007/04/01 (d), shell height (cm)
44.392	4.366
130.709	4.629
169.347	4.782
203.052	4.967
246.622	5.153
289.369	5.252
336.227	5.307
373.221	5.352
401.171	5.418
434.054	5.538
462.005	5.582
485.023	5.669
554.077	5.856];
units.tL215   = {'d', 'cm'};  label.tL215 = {'time', 'shell height', 'grp 2, 15 m'};  
temp.tL215    = C2K(3);  units.temp.tL215 = 'K'; label.temp.tL215 = 'temperature';
bibkey.tL215 = 'BlicRysg2010';
comment.tL215 = 'Data for grp 2, 15 m deep';
% grp 2, 30 m deep
data.tL230 = [ ... % time since 2007/04/01 (d), dry weight (g)
43.570	4.496
131.532	4.748
170.169	4.901
203.052	5.010
246.622	5.109
290.191	5.219
337.050	5.264
373.221	5.352
401.171	5.396
434.876	5.440
462.005	5.517
485.023	5.561
554.077	5.791];
units.tL230   = {'d', 'cm'};  label.tL230 = {'time', 'shell height', 'grp 2, 30 m'};  
temp.tL230    = C2K(2);  units.temp.tL230 = 'K'; label.temp.tL230 = 'temperature';
bibkey.tL230 = 'BlicRysg2010';
comment.tL230 = 'Data for grp 2, 30 m deep';
%
% time-dry weight
% grp 1, 15 m deep
data.tW115 = [ ... % time since 2007/04/01 (d), dry weight (g)
44.572	0.171
131.529	0.411
168.299	0.557
202.867	0.633
246.411	0.608
288.793	0.684
336.792	0.677
371.381	0.709
400.303	0.930
433.724	1.076
458.258	1.127
480.475	1.367
551.777	1.671];
units.tW115   = {'d', 'g'};  label.tW115 = {'time', 'dry weight', 'grp 1, 15 m'};  
temp.tW115    = C2K(2);  units.temp.tW115 = 'K'; label.temp.tW115 = 'temperature';
bibkey.tW115 = 'BlicRysg2010';
comment.tW115 = 'Data for grp 1, 15 m deep';
% grp 1, 30 m deep
data.tW130 = [ ... % time since 2007/04/01 (d), dry weight (g)
44.580	0.152
130.435	0.361
169.474	0.424
202.952	0.443
245.362	0.456
289.996	0.487
335.781	0.443
372.579	0.525
400.425	0.658
432.713	0.842
460.601	0.880
481.772	0.962
550.932	1.063];
units.tW130   = {'d', 'g'};  label.tW130 = {'time', 'dry weight', 'grp 1, 30 m'};  
temp.tW130    = C2K(2);  units.temp.tW130 = 'K'; label.temp.tW130 = 'temperature';
bibkey.tW130 = 'BlicRysg2010';
comment.tW130 = 'Data for grp 1, 30 m deep';
% grp 2, 15 m deep
data.tW215 = [ ... % time since 2007/04/01 (d), dry weight (g)
43.399	0.873
129.085	1.570
168.034	2.038
202.530	2.089
244.817	2.101
288.216	2.114
333.841	2.038
370.564	2.177
398.384	2.620
431.768	3.025
457.363	2.785
480.732	2.886
549.726	3.747];
units.tW215   = {'d', 'g'};  label.tW215 = {'time', 'dry weight', 'grp 2, 15 m'};  
temp.tW215    = C2K(2);  units.temp.tW215 = 'K'; label.temp.tW215 = 'temperature';
bibkey.tW215 = 'BlicRysg2010';
comment.tW215 = 'Data for grp 2, 15 m deep';
% grp 2, 30 m deep
data.tW230 = [ ... % time since 2007/04/01 (d), dry weight (g)
44.512	0.949
131.311	1.633
169.146	1.810
202.530	1.646
244.817	1.759
289.329	1.722
333.841	1.671
370.564	1.848
398.384	2.025
432.881	2.582
458.476	2.532
480.732	2.443
550.838	3.038];
units.tW230   = {'d', 'g'};  label.tW230 = {'time', 'dry weight', 'grp 2, 30 m'};  
temp.tW230    = C2K(2);  units.temp.tW230 = 'K'; label.temp.tW230 = 'temperature';
bibkey.tW230 = 'BlicRysg2010';
comment.tW230 = 'Data for grp 2, 30 m deep';

%% set weights for all real data
weights = setweights(data, []);

%% set pseudodata and respective weights
[data, units, label, weights] = addpseudodata(data, units, label, weights);

%% pack auxData and txtData for output
auxData.temp = temp;
txtData.units = units;
txtData.label = label;
txtData.bibkey = bibkey;
txtData.comment = comment;

%% Group plots
set1 = {'tL115','tL130','tL215','tL230'}; subtitle1 = {'Data for grp 1, 15 & 30 m; grp 2, 15 & 30 m'};
set2 = {'tW115','tW130','tW215','tW230'}; subtitle2 = {'Data for grp 1, 15 & 30 m; grp 2, 15 & 30 m'};
metaData.grp.sets = {set1,set2};
metaData.grp.subtitle = {subtitle1,subtitle2};

%% Acknowledgment
metaData.acknowledgment = 'The creation of this entry was supported by the Norwegian Science Council (NFR 255295)';

%% Links
metaData.links.id_CoL = '69X4D'; % Cat of Life
metaData.links.id_ITIS = '79619'; % ITIS
metaData.links.id_EoL = '46467714'; % Ency of Life
metaData.links.id_Wiki = 'Chlamys_islandica'; % Wikipedia
metaData.links.id_ADW = 'Chlamys_islandica'; % ADW
metaData.links.id_Taxo = '39391'; % Taxonomicon
metaData.links.id_WoRMS = '140692'; % WoRMS
metaData.links.id_molluscabase = '140692'; % molluscabase


%% References
bibkey = 'Wiki'; type = 'Misc'; bib = ...
'howpublished = {\url{http://en.wikipedia.org/wiki/Chlamys_islandica}}';
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
bibkey = 'BlicRysg2010'; type = 'Article'; bib = [ ... 
'author = {M. E. Blicher and S. Rysgaard and M. K. Sejr}, ' ... 
'year = {2013}, ' ...
'title = {Seasonal growth variation in \emph{Chlamys islandica} (Bivalvia) from sub-{A}rctic {G}reenland is linked to food availability and temperature}, ' ...
'journal = {Mar. Ecol. Prog. Ser.}, ' ...
'volume = {407}, ' ...
'pages = {71-86}'];
metaData.biblist.(bibkey) = ['''@', type, '{', bibkey, ', ' bib, '}'';'];
%
bibkey = 'ShumPars2016'; type = 'Book'; bib = [ ...  % used in setting of chemical parameters and pseudodata
'author = {Shumway, S. E. and Parsons, G. J. (eds)}, ' ...
'year = {2016}, ' ...
'title  = {Scallops: biology, ecology, aquaculture, and fisheries}, ' ...
'publisher = {Elsevier}, ' ...
'series = {Developments in aquaculture and fisheries science}, ' ...
'volume = {40}, ' ...
'howpublished = {\url{../../../bib/Kooy2010.html}}'];
metaData.biblist.(bibkey) = ['''@', type, '{', bibkey, ', ' bib, '}'';'];
%
bibkey = 'sealifebase'; type = 'Misc'; bib = ...
'howpublished = {\url{http://www.sealifebase.org/summary/Chlamys-islandica.html}}';
metaData.biblist.(bibkey) = ['''@', type, '{', bibkey, ', ' bib, '}'';'];

