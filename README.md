# TFG – Arquitectura de computadors i sistemes operatius

## Definició dels objectius del TFG de forma clara i concreta. Planificació temporal.

Estudiant: Jordi Bericat Ruz
Professor col·laborador: Daniel Rivas Barragan
Semestre: Tardor 2021/22 (Aula 1)

## Índex

1 - Títol provisional del projecte	
2 - Resum provisional del projecte: Motivació i definició dels objectius	
3 - Planificació temporal	
3.1 – Descripció de cada fita i detall de tasques relacionades	
3.2 – Planificació temporal per fites (Google Calendar)	
 
## 1 - Títol provisional del projecte

**Simulació d’un conjunt de mitjans aeris específics per a la extinció d’incendis forestals durant períodes nocturns i dirigits de manera autònoma mitjançant tècniques d’intel·ligència artificial.**

## 2 - Resum provisional del projecte: Motivació i definició dels objectius

Durant els darrers 35 anys , s’ha incrementat tant la envergadura com la virulència dels incendis forestals que es produeixen a Catalunya: hi ha menys incendis, però aquests són molt més grans i avancen més ràpidament. La causa s’atribueix a diversos factors; principalment al progressiu escalfament global del planeta, que està afectant especialment les zones costaneres del mar Mediterrani  i que implica que es produeixin situacions de sequera no només a l’estiu, si no que també en d’altres èpoques de l’any, però també a l’abandonament de les terres de conreu i pastures, agreujat per la ineficient política de gestió dels boscos del nostre país (gairebé un 65%  del territori català correspon a massa forestal). 

Aquesta nova categoria de grans incendis, sumada a la evolució accelerada del sector tecnològic durant els darrers anys; com és la invenció de les bateries de liti , les quals estan permetent la fabricació de drons de darrera generació amb alta capacitat de càrrega  (i molt més econòmics que els mitjans aeris d’extinció d’incendis actuals), així com als avenços en matèria d’intel·ligència artificial i visió per computador , que permetrien que aquests nous drons funcionessin pràcticament de manera autònoma, podrien ser els factors clau que impulsessin el desenvolupament de noves tecnologies per al seu ús en la lluita aèria contra incendis forestals, que sense pretendre substituir el mitjans aeris actuals a curt o mig termini, sí que podrien esdevenir eines de suport en aquells escenaris en què els mitjans comunament utilitzats avui en dia no siguin capaços d’actuar. 

En aquest marc és on vaig trobar la motivació que em va dur a pensar en la idea inicial de desenvolupar un sistema coordinat o eixam de drons “CAV”   controlats de manera autònoma mitjançant tecnologies basades en visió per computador i xarxes neuronals artificials, específic per al suport aeri en tasques d’extinció d’incendis forestals durant períodes nocturns (quan es dona el cas què els medis aeris tradicionals d’extinció d’incendis  no poden treballar degut a les reduïdes condicions de visibilitat, però que suposaria el millor moment per actuar, donada la millora de les condicions climatològiques pel que respecta la baixada de la temperatura i la pujada de la humitat).
 
Tanmateix, sóc plenament conscient de què, avui en dia, a nivell de hardware encara no existeix cap producte comercial amb les capacitats tècniques necessàries per assolir aquests objectius, tot i que sí que hi ha empreses / organitzacions  que estan treballant en aquest sentit i que ja han desenvolupat alguns prototips de dron CAV que permetrien, en un futur no molt llunyà, la implementació d’aquesta idea de projecte. De la mateixa manera, també tinc present que el desenvolupament complert d’un projecte d’aquestes característiques resta fora de l’abast d’un TFG. Així doncs, per a la realització del mateix em centraré en dissenyar una Proof of Concept (PoC) que permeti demostrar, mitjançant la implementació limitada a uns casos d’ús concrets, que la meva idea és factible a nivell de software. 

En aquest sentit, d’entrada un dels objectius del TFG estarà orientat a establir un entorn virtual mitjançant la plataforma de simulació de vehicles autònoms anomenada AirSim® , desenvolupada en codi obert per la companyia Microsoft, que és específica per a la recerca en l’àrea de AI / DL i que està implementada utilitzant el framework de gràfics 3D Unreal Engine. Per assolir aquest objectiu es farà ús de les API per al llenguatge de programació Python que la mateixa plataforma de simulació AirSim proporciona. Un cop assolit, procediré, o bé a dissenyar i implementar una arquitectura de xarxes neuronals convolucionals profundes (Deep Convolutional Neural Network o DCNN), o bé a adaptar-ne una de ja existent, amb la qual serà possible entrenar els vehicles dron simulats per a què realitzin una sèrie d’accions concretes a mode de PoC. A partir d’aquest punt ja disposaré dels mitjans que em permetran demostrar que, avui en dia, alguns dels aspectes de la meva idea inicial són assolibles a nivell de software. Concretament:
1.	Mitjançant l’arquitectura DCNN dissenyada, demostraré que el col·lectiu de drons  pertanyents a l’eixam són capaços, durant una fase inicial d’entrenament, d’aprendre les característiques d’un incendi forestal simulat en condicions de foscor. A més, també quedarà demostrat que podran utilitzar aquest aprenentatge durant una segona fase d’explotació amb l’objectiu d’identificar i classificar les zones afectades per altres instàncies d’incendi, diferents a les reconegudes durant la fase d’aprenentatge.
2.	Demostraré que els drons de l’eixam poden establir la comunicació necessària entre ells per tal de coordinar-se sobre el terreny de manera autònoma. La esmentada comunicació vindrà determinada per la informació que cada dron pugui recol·lectar de manera individual, en funció de les característiques de l’incendi forestal concret.

A tall de resum; el sistema estarà dissenyat per a ser capaç, d’una banda, d’identificar i classificar de manera totalment autònoma les característiques d’un incendi amb una configuració diferent de les que ha aprés l’arquitectura DCNN durant la fase d’entrenament, i de l’altra, de decidir en temps real i de manera distribuïda en quines zones serà necessari actuar primer, de manera que serà possible repartir la càrrega de feina o workload entre els diferents drons de l’eixam. Ho podem il·lustrar de manera pràctica si posem el cas que, durant un deployment de l’eixam durant la nit, cadascun dels drons està tractant de refredar diferents parts del perímetre d’un incendi amb l’objectiu d’evitar una revifada de les flames, que ja han sigut extingides durant horari diürn pels mitjans aeris tradicionals (e.g. hidroavions i helitancs). En aquest escenari, si suposem que un dels drons conclou que s’està produint una revifada a un dels fronts de l’incendi, aleshores, aquest dron pot comunicar-ho a l’eixam sencer per a què s’avaluï de manera distribuïda si la resta de drons han de deixar estar la tasca que estan realitzant i dirigir-se cap a la seva zona per tal de donar-li suport.
 
## 3 - Planificació temporal

### 3.1 – Descripció de cada fita i detall de tasques relacionades

1. Fita #001: Establir i preparar l’entorn de desenvolupament 
1.1. Descripció i preparació / muntatge del hardware disponible 
1.2. Instal·lació i configuració del Sistema Operatiu Ubuntu Linux 18.04 LTS
1.3. Instal·lació i configuració de la plataforma de simulació AirSim de Microsoft i dels components necessaris (LandscapeMountain.zip Environment Asset Pack)
1.4. Instal·lació i configuració del framework de gràfics 3D Unreal Engine 4.xx  i dels components necessaris (Fire Builder Asset Pack)
1.5. Instal·lació del IDE de desenvolupament PyDev basat en Eclipse, així com dels frameworks necessaris per a implementar la DCNN (o bé Keras, o bé Tensor-Flow)
1.6. Instal·lació del client Git per a Linux i creació del repositori al GitHub 
2. Fita #002: Adaptació de la plataforma de simulació de vehicles autònoms AirSim de Microsoft 
2.1. Simulació de l’escenari (terreny)
2.1.1. Establir el terreny apropiat: Massa forestal i/o terreny muntanyós mitjançant el “Landscape environment pack” de AirSim
2.1.2. Generar diferents escenaris amb incendis de manera aleatòria utilitzant el Fire Builder Asset Pack del framework Unreal Engine
2.1.3. Simular un escenari nocturn mitjançant l’arranjament de paràmetres nadius de AirSim
2.2. Simulació dels actors (5 drons i/o octocòpters) mitjançant la API de AirSim 
3. Fita #003: O bé dissenyar i implementar l’arquitectura DCNN mitjançant els frameworks Keras o Tensor-Flow per a Python, o bé adaptar-ne una de ja existent en funció dels casos d’ús. En tot cas: 
3.1. Primer definirem les classes de la capa de sortida o completa de l’arquitectura DCNN
3.2. Després cal determinar quines característiques o features s’hauran de reconèixer en funció de les classes que haguem determinat al punt anterior
3.3. Seguidament es decidirà de quina manera (mida, colors, etc) es normalitzarà el training data-set d’imatges per tal de conèixer de partida la complexitat computacional i de memòria a la que haurem de fer front, i s’adaptarà el data-set en conseqüència
3.4. Establir quins filtres o Kernels necessitarem aplicar a les deep / hidden layers de la DCNN
3.5. Acabar de depurar / definir la resta de capes que ha d’incloure l’arquitectura (pools, etc.) 
4. Fita #004: Generar training data-set que servirà per a entrenar la DCNN (segons els paràmetres establerts al punt 3) amb imatges virtuals d’incendis obtingudes del rendering generat des de la mateixa plataforma AirSim 
5. Fita #005: Entrenar la DCNN mitjançant el training data-set apropiadament normalitzat
6. Fita #006: Testeig de la DCNN (Joc de proves #1):
6.1. Definir un joc de proves reduït per a testar el correcte funcionament de la DCNN amb un sol dron
6.2. Entrenar i fer el deployment d’un sol dron sobre el joc de proves #1
6.3. Avaluar el comportament de la DCNN: En cas de fallada repassar el punt 4, o bé repetir el punt 5 amb un training data-set amb un conjunt d’imatges més gran. En cas d’èxit seguir amb el punt 7)
7. Fita #007: Entrenar la resta de drons de l’eixam (5 unitats de drons o octocòpters)
8. Fita #008: Disseny + implementació dels mecanismes de comunicació entre els drons de l’eixam:
8.1. Definir els mecanismes de comunicació que utilitzaran els drons tant pel que fa la rebuda de comandes de control com pel que respecta al pas de missatges a la resta de l’eixam 
8.2. Descriure amb més detall les PoC que prendrem com a referència per a implementar la comunicació: quines ordres es comuniquen, quines dades s’envien (e.g. coordenades GPS) i quines accions s’efectuen
8.3. Definir un sistema de decisió de prioritat dels missatges inter-dron, així com establir com es gestiona el cas en què dos o més missatges amb la mateixa prioritat arribin “a l’hora”. A tall d’exemple; si dos drons envien un mateix missatge a l’hora es determina quin és el de més prioritat i s’actua en conseqüència, però si tenen la mateixa prioritat, aleshores per a determinar quina ordre s’accepta o es descarta s’haurà d’utilitzar un mecanisme de decisió distribuït; o bé un de senzill basat en timestamps en cas que els drons (nodes) de l’eixam estiguin perfectament sincronitzats, o bé, en cas de que no ho estiguin, utilitzar altres mecanismes més complexos com per exemple Lamport timestamp o bé Vector Clocks. 
8.4. Definir quin mecanisme (distribuït) s’utilitza per a propagar missatges per tot l’eixam. p.e. Gossip Protocol, Time-Stamped Anti-Entropy Protocol (TSAE), etc.
8.5. Implementar els mecanismes de comunicació distribuïda que s’hagi decidit utilitzar
9. Fita #009: Testeig de les comunicacions (joc de proves #2): 
9.1. Definir un joc de proves per tal de comprovar que la comunicació inter-dron s’efectua de manera correcta
9.2. Executar el joc de proves sobre tot l’eixam
9.3. Avaluar els resultats del testeig. Si falla tornar al punt 8, si no, passar al punt 10.
10. Fita #010: Testeig de tot el sistema (joc de proves #3):
10.1. Unir i estendre els jocs de proves #1 i #2 per a poder provar els casos d’ús (PoC) definits  a l’abstract del projecte.  
10.2. Fer el Deployment de tot l’eixam sobre el joc de proves nº 3
10.3. Avaluar resultats i procedir en conseqüència
11. Fita #011: Obtenir conclusions i redactar la memòria del projecte
12. Fita #012: Crear una presentació i/o vídeo-demo del projecte que destaqui els punts principals i demostri el correcte funcionament de tot el sistema sobre les PoC definides.
13. Fita #013: Preparar la defensa del projecte

### 3.2 – Planificació temporal per fites (Google Calendar)

Seguidament es detalla la planificació temporal de cadascuna de les fites establertes a la secció 3.1. El mateix calendari pot ser consultat a la aplicació web “Google Calendar” mitjançant el següent enllaç, de manera que serà possible fer un seguiment de l’assoliment de cada fita per part del professor docent col·laborador de l’assignatura, si això ho creu convenient:


https://calendar.google.com/calendar/u/1?cid=Y19lc3VvZnFqMmM1NGJsMmM0NTJ1b3VvMnA0MEBncm91cC5jYWxlbmRhci5nb29nbGUuY29t 

4 - Bibliografia

Salvatierra, Issac; Bosch, Francina; Marfà, Ricard; Longan, Idoia. Per què Catalunya crema [Internet]. Barcelona: [Actualitzat el (N/A) ; consultat l’1 d’octubre de 2021]. Disponible a:  https://interactius.ara.cat/incendis/

Elgendy, Mohamed. Deep Learning for Vision Systems. Nova York, Estats Units d’Amèrica: Manning publications; 2020. 458 p.

Yadav, Robin. Deep Learning Based Fire Recognition for Wildfire Drone Automation. The Canadian Science Fair Journal. 2020 Oct;3(2):N/A. https://csfjournal.com/volume-3-issue-2/2020/10/30/deep-learning-based-fire-recognition-for-wildfire-drone-automation

Microsoft Research. Aerial Informatics and Robotics Platform [Internet]. Microsoft: Microsoft Research; 2017 [Actualitzat el 5 de febrer de 2021; consultat el l’9 d’octubre de 2021]. https://www.microsoft.com/en-us/research/project/aerial-informatics-robotics-platform/#overview

Universitat Oberta de Catalunya. Com citar: Estil Vancouver [Internet]. Campus virtual: Recursos d’aprenentatge de l’assignatura “TFG – Arquitectura de computadors i sistemes operatius”; [Actualitzat el (N/A); consultat el 9 d’octubre de 2021]. Disponible a:  https://biblioteca.uoc.edu:8080/ca/plana/Estil-Vancouver/ 

Beneito Montagut, Roser. Presentació de documents i elaboració de presentacions [Internet]. Barcelona: Editorial UOC; data de publicació no disponible [consultat el 9 d’octubre de 2021]. 56 p. Disponible a: https://campus.uoc.edu/cdocent/HKZB5UG6XS130_6R5O43.pdf 


## *changelog:*

v001.1 - Afegida documentació de tasques relacionades  amb la fita #001 (Preparació de l'entorn) - 
         /doc/Fita #001 - Preparació de l'entorn_ESBORRANY.pdf
