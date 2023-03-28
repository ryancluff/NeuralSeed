/*
model : SimpleRNN
input_size : 1
skip : 1
output_size : 1
unit_type : LSTM
num_layers : 1
hidden_size : 8
bias_fl : True
*/

std::vector<std::vector<float>> rec_weight_ih_l0 = {{0.040201589465141296, 0.04993141442537308, 0.19460153579711914, -0.41107985377311707, -0.024106381461024284, -0.2172476351261139, 0.025126514956355095, 0.5169431567192078, 0.30458924174308777, 0.1784122884273529, 0.31854239106178284, -0.5279170274734497, 0.30707013607025146, -0.06234700232744217, -0.009385074488818645, 0.569477915763855, -1.613985538482666, 0.8607062101364136, 0.9982549548149109, -0.504291296005249, -0.3038516938686371, 2.257394552230835, 0.7041603326797485, 0.01963549107313156, 0.09572724252939224, 0.20740927755832672, 0.5308672785758972, -0.465284138917923, 0.26188114285469055, -0.19679147005081177, 0.03066636249423027, 0.3567180037498474}}; 

std::vector<std::vector<float>> rec_weight_hh_l0 = {{0.48196473717689514, -0.02755555510520935, -0.05371885001659393, -0.2222432643175125, 0.07121361792087555, 0.08817927539348602, -0.1352425217628479, 0.3104941248893738, 0.21688872575759888, 0.09155268222093582, -0.14575031399726868, 0.12829472124576569, 0.1678198128938675, 0.15280844271183014, -0.08847618848085403, 0.2014641910791397, 1.3999004364013672, 0.2722837030887604, -0.10111255198717117, 0.2526962161064148, 0.06263182312250137, 0.05682223662734032, 0.09424073249101639, 0.539299488067627, 0.5197988152503967, -0.019363516941666603, 0.14695462584495544, 0.06763333082199097, 0.11094015836715698, -0.011093615554273129, -0.12494396418333054, 0.25778695940971375}, 
                                                    { 0.08931037783622742, -0.21020928025245667, -0.025001905858516693, 0.14614978432655334, -0.109410360455513, 0.1587153822183609, 0.05106334388256073, -0.1573590636253357, -0.12411394715309143, -0.06781285256147385, -0.13231724500656128, 0.18669483065605164, 0.05425984412431717, 0.058632154017686844, 0.07721223682165146, -0.30143025517463684, -0.16818355023860931, 0.41216060519218445, 0.4420204758644104, -0.2869819700717926, 0.08285996317863464, -0.03890977427363396, -0.4783587157726288, -0.23163805902004242, 0.19261814653873444, 0.21198028326034546, -0.09583200514316559, 0.13498158752918243, 0.05070319026708603, 0.1584230363368988, 0.04141049087047577, -0.036011263728141785}, 
                                                    { 0.10854537785053253, -0.2652236819267273, 0.10393232107162476, -0.07098986953496933, -0.10215991735458374, 0.04473597928881645, 0.11148901283740997, 0.19361832737922668, -0.1140144094824791, -0.06907657533884048, 0.09295305609703064, -0.0042410921305418015, 0.5985769033432007, -0.07342834025621414, 0.021796833723783493, 0.2737938165664673, -0.1081412211060524, -0.4624134302139282, -0.21372175216674805, -0.1317746639251709, -0.3058769106864929, 0.38064053654670715, 0.015032656490802765, -0.38387995958328247, 0.015707990154623985, -0.029715944081544876, 0.17315155267715454, -0.10090755671262741, 0.16295811533927917, 0.03638701140880585, 0.10559846460819244, 0.2390775829553604}, 
                                                    { -0.06423600763082504, 0.5779485106468201, -0.06401999294757843, 0.19901016354560852, 0.2860023081302643, -0.06429802626371384, -0.022145746275782585, 0.22353699803352356, -0.2314385622739792, 0.4087222218513489, -0.1222718358039856, 0.4445403814315796, 0.7949437499046326, -0.20773908495903015, -0.09828965365886688, 0.091793954372406, -0.30348995327949524, 0.6457017660140991, -0.3697372376918793, 0.032841481268405914, 0.33501097559928894, 0.3013899028301239, 0.21127437055110931, 0.09425278753042221, -0.06378233432769775, 0.1263430416584015, -0.06358558684587479, 0.39925694465637207, 0.8545874953269958, -0.049455996602773666, -0.01377861574292183, 0.08967088907957077}, 
                                                    { -0.007710456382483244, -0.07488512992858887, 0.12793320417404175, 0.06043344736099243, -0.17139829695224762, 0.08637899160385132, -0.008564375340938568, 0.10413168370723724, -0.35658198595046997, -0.20367756485939026, -0.032779574394226074, 0.2738420367240906, 0.4079212546348572, -0.07585665583610535, -0.041628047823905945, -0.3090572953224182, -1.9761296510696411, -0.5584377646446228, 0.012536442838609219, -0.05497186630964279, 0.5261538624763489, 0.7877647876739502, 0.8902642726898193, 0.4571446180343628, 0.22631265223026276, 0.08969278633594513, -0.024054739624261856, 0.3333570659160614, 0.1498458981513977, 0.11317908018827438, -0.006161061115562916, -0.16441728174686432}, 
                                                    { 0.03475503250956535, -0.010218382813036442, -0.12130102515220642, -0.2614767849445343, 0.09631065279245377, -0.051828477531671524, -0.019963180646300316, 0.21433696150779724, 0.33879682421684265, -0.029239434748888016, -0.15290865302085876, -0.28000152111053467, -0.3833125829696655, -0.10433385521173477, -0.028499074280261993, 0.24537840485572815, -0.88942950963974, -0.04625876247882843, -0.33736419677734375, -0.365448534488678, -0.2997126579284668, -0.07735907286405563, 1.6335606575012207, -0.10645139962434769, -0.012208530679345131, -0.13058270514011383, -0.05948976054787636, -0.4179331660270691, 0.11541227251291275, -0.1272607445716858, -0.07476994395256042, 0.3030123710632324}, 
                                                    { -0.36611971259117126, -0.04879450425505638, 0.18547117710113525, -0.2370576411485672, -0.04629778489470482, 0.09907568991184235, -0.07520247995853424, 0.18606457114219666, -0.048406630754470825, 0.1687690019607544, 0.015305493026971817, -0.485907644033432, -0.3745999336242676, 0.09610747545957565, 0.03941512107849121, 0.06330476701259613, -1.4506404399871826, 0.2831016182899475, 0.13357412815093994, -0.02797056920826435, -0.3080710768699646, -0.3098300099372864, -0.44418877363204956, -0.6129707098007202, -0.8239648938179016, 0.02779342606663704, 0.14953647553920746, -0.3344722092151642, -0.09844790399074554, 0.017136523500084877, -0.10739048570394516, 0.053050920367240906}, 
                                                    { 0.0947142243385315, -0.4045710265636444, -0.023112474009394646, 0.028233034536242485, -0.46688491106033325, 0.15146248042583466, -0.017030710354447365, -0.16576245427131653, 0.19571158289909363, 0.00887699332088232, -0.2628317177295685, 0.04022897779941559, -0.5113086104393005, 0.15239018201828003, -0.05645909905433655, -0.6051189303398132, -0.5008705854415894, 0.8597032427787781, -0.1959954798221588, 0.13800127804279327, 0.40599361062049866, 0.3820352256298065, 1.5358766317367554, -0.142281636595726, 0.30420994758605957, -0.09352054446935654, -0.18632827699184418, 0.109166719019413, -0.41232815384864807, 0.12900304794311523, -0.00877341814339161, -0.2884884476661682}}; 

std::vector<std::vector<float>> lin_weight = {{0.0038844842929393053, -1.1155909299850464, -1.0957703590393066, 0.19735965132713318, -0.6868059039115906, -0.4445183277130127, 0.11942995339632034, 0.010442369617521763}}; 

std::vector<float> lin_bias = {0.14359237253665924}; 

std::vector<float> lstm_bias_sum = {2.4189088344573975, -1.8546819686889648, 0.15531245805323124, -0.6672953069210052, -2.460897922515869, 1.7038464546203613, 2.039452075958252, -0.15579792112112045, -0.23863479495048523, 2.3136411905288696, -0.10608039796352386, 0.1709616631269455, 2.3506157398223877, -0.47845594584941864, -0.067105358466506, -0.027078092098236084, -0.10537301376461983, 0.01665624976158142, 0.1615859717130661, -0.019800350069999695, -0.03463376313447952, -0.10172984376549721, -0.12552976608276367, 0.07326128333806992, 1.5116127133369446, 0.6343513131141663, 0.1989677380770445, 0.15540845319628716, 1.1703882813453674, 1.8854008913040161, 2.2120660543441772, 1.1053834557533264}; 
