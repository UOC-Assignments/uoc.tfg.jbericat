# Universitat Oberta de Catalunya - Treball Final de Grau 
# *Arquitectura de computadors i sistemes operatius*

Estudiant: Jordi Bericat Ruz

Professor col·laborador: Daniel Rivas Barragan

Semestre: Tardor 2021/22 (Aula 1)

## Índex

1. [Títol provisional del projecte](#1---t%C3%ADtol-provisional-del-projecte)

2. [Resum provisional del projecte: Motivació i definició dels objectius](#2---resum-provisional-del-projecte-motivaci%C3%B3-i-definici%C3%B3-dels-objectius)

3. [Definició dels objectius del TFG: Planificació temporal](#3---planificaci%C3%B3-temporal)

    3.1. [Descripció de cada fita i detall de tasques relacionades](#31--planificaci%C3%B3-temporal-per-fites-google-calendar)

    3.2. [Planificació temporal per fites (Google Calendar)](#32--descripci%C3%B3-de-cada-fita-i-detall-de-tasques-relacionades)

4. [Bibliografia](#4---bibliografia)
 
## 1 - Títol provisional del projecte

**Entrenament d’un model d'intel·ligència artificial en un entorn virtual per a la seva aplicació en la extinció d’incendis forestals: Prova de concepte**

## 2 - Resum provisional del projecte: Motivació i definició dels objectius

Durant els darrers 35 anys , s’ha incrementat tant la envergadura com la virulència dels incendis forestals que es produeixen a Catalunya: hi ha menys incendis, però aquests són molt més grans i avancen més ràpidament. La causa s’atribueix a diversos factors; principalment al progressiu escalfament global del planeta, que està afectant especialment les zones costaneres del mar Mediterrani  i que implica que es produeixin situacions de sequera no només a l’estiu, si no que també en d’altres èpoques de l’any, però també a l’abandonament de les terres de conreu i pastures, agreujat per la ineficient política de gestió dels boscos del nostre país (gairebé un 65%  del territori català correspon a massa forestal). 

Aquesta nova categoria de grans incendis, sumada a la evolució accelerada del sector tecnològic durant els darrers anys; com és la invenció de les bateries de liti , les quals estan permetent la fabricació de drons de darrera generació amb alta capacitat de càrrega  (i molt més econòmics que els mitjans aeris d’extinció d’incendis actuals), així com als avenços en matèria d’intel·ligència artificial i visió per computador , que permetrien que aquests nous drons funcionessin pràcticament de manera autònoma, podrien ser els factors clau que impulsessin el desenvolupament de noves tecnologies per al seu ús en la lluita aèria contra incendis forestals, que sense pretendre substituir el mitjans aeris actuals a curt o mig termini, sí que podrien esdevenir eines de suport en aquells escenaris en què els mitjans comunament utilitzats avui en dia no siguin capaços d’actuar. 

En aquest marc és on vaig trobar la motivació que em va dur a pensar en la idea inicial de desenvolupar un sistema coordinat o eixam de drons “CAV”   controlats de manera autònoma mitjançant tecnologies basades en visió per computador i xarxes neuronals artificials, específic per al suport aeri en tasques d’extinció d’incendis forestals durant períodes nocturns (quan es dona el cas què els medis aeris tradicionals d’extinció d’incendis  no poden treballar degut a les reduïdes condicions de visibilitat, però que suposaria el millor moment per actuar, donada la millora de les condicions climatològiques pel que respecta la baixada de la temperatura i la pujada de la humitat).
 
Tanmateix, sóc plenament conscient de què, avui en dia, a nivell de hardware encara no existeix cap producte comercial amb les capacitats tècniques necessàries per assolir aquests objectius, tot i que sí que hi ha empreses / organitzacions  que estan treballant en aquest sentit i que ja han desenvolupat alguns prototips de dron CAV que permetrien, en un futur no molt llunyà, la implementació d’aquesta idea de projecte. De la mateixa manera, també tinc present que el desenvolupament complert d’un projecte d’aquestes característiques resta fora de l’abast d’un TFG. Així doncs, per a la realització del mateix em centraré en dissenyar una Proof of Concept (PoC) que permeti demostrar, mitjançant la implementació limitada a uns casos d’ús concrets, que la meva idea és factible a nivell de software. 

En aquest sentit, d’entrada un dels objectius del TFG estarà orientat a establir un entorn virtual mitjançant la plataforma de simulació de vehicles autònoms anomenada AirSim® , desenvolupada en codi obert per la companyia Microsoft, que és específica per a la recerca en l’àrea de AI / DL i que està implementada utilitzant el framework de gràfics 3D Unreal Engine. Per assolir aquest objectiu es farà ús de les API per al llenguatge de programació Python que la mateixa plataforma de simulació AirSim proporciona. Un cop assolit, procediré, o bé a dissenyar i implementar una arquitectura de xarxes neuronals convolucionals profundes (Deep Convolutional Neural Network o DCNN), o bé a adaptar-ne una de ja existent, amb la qual serà possible entrenar els vehicles dron simulats per a què realitzin una sèrie d’accions concretes a mode de PoC. A partir d’aquest punt ja disposaré dels mitjans que em permetran demostrar que, avui en dia, alguns dels aspectes de la meva idea inicial són assolibles a nivell de software. Concretament:
1.	Mitjançant l’arquitectura DCNN dissenyada, demostraré que el col·lectiu de drons  pertanyents a l’eixam són capaços, durant una fase inicial d’entrenament, d’aprendre les característiques d’un incendi forestal simulat en condicions de foscor. A més, també quedarà demostrat que podran utilitzar aquest aprenentatge durant una segona fase d’explotació amb l’objectiu d’identificar i classificar les zones afectades per altres instàncies d’incendi, diferents a les reconegudes durant la fase d’aprenentatge.
2.	Demostraré que els drons de l’eixam poden establir la comunicació necessària entre ells per tal de coordinar-se sobre el terreny de manera autònoma. La esmentada comunicació vindrà determinada per la informació que cada dron pugui recol·lectar de manera individual, en funció de les característiques de l’incendi forestal concret.

A tall de resum; el sistema estarà dissenyat per a ser capaç, d’una banda, d’identificar i classificar de manera totalment autònoma les característiques d’un incendi amb una configuració diferent de les que ha aprés l’arquitectura DCNN durant la fase d’entrenament, i de l’altra, de decidir en temps real i de manera distribuïda en quines zones serà necessari actuar primer, de manera que serà possible repartir la càrrega de feina o workload entre els diferents drons de l’eixam. Ho podem il·lustrar de manera pràctica si posem el cas que, durant un deployment de l’eixam durant la nit, cadascun dels drons està tractant de refredar diferents parts del perímetre d’un incendi amb l’objectiu d’evitar una revifada de les flames, que ja han sigut extingides durant horari diürn pels mitjans aeris tradicionals (e.g. hidroavions i helitancs). En aquest escenari, si suposem que un dels drons conclou que s’està produint una revifada a un dels fronts de l’incendi, aleshores, aquest dron pot comunicar-ho a l’eixam sencer per a què s’avaluï de manera distribuïda si la resta de drons han de deixar estar la tasca que estan realitzant i dirigir-se cap a la seva zona per tal de donar-li suport.
 
## 3 - Definició dels objectius del TFG: Planificació temporal

### 3.1 – Planificació temporal per fites (Google Calendar)

El calendari de planificació pot ser consultat a la aplicació web “Google Calendar” mitjançant el següent enllaç:

https://calendar.google.com/calendar/u/1?cid=Y19lc3VvZnFqMmM1NGJsMmM0NTJ1b3VvMnA0MEBncm91cC5jYWxlbmRhci5nb29nbGUuY29t 

Seguidament es detalla la planificació temporal de cadascuna de les tasques associades a cada fita.

### 3.2 – Descripció de cada fita i detall de tasques relacionades

    1. FITA#01 - Establir i preparar l’entorn de desenvolupament
 
        1.1. Anàlisi de pre-requisits

        1.2. Característiques del maquinari

        1.3. Preparació del programari

            1.3.1 Selecció i instal·lació del sistema operatiu (Linux Workstation)

            1.3.2. Estructura de directoris del projecte

            1.3.3. Instal·lació i configuració dels paquets de software

                1.3.3.1. Paquets base i dependències

                1.3.3.2. Controladors de dispositiu

                1.3.3.3. IDE “Visual Studio Code”

                1.3.3.4. Motor gràfic: Unreal Engine

                1.3.3.5. Plataforma de simulació: AirSim (Aerial Informatics and Robotics Simulation)

                1.3.3.6. Paquet d’entorn LandscapeMountains per al motor gràfic Unreal Engine

                1.3.3.7. Paquet d’assets M5VFX vol.2 per al motor gràfic Unreal Engine 

    2. FITA#02 - Adaptació i personalització de l’entorn de la plataforma de simulació de vehicles autònoms AirSim de Microsoft

        2.1. Obtenir l’entorn Landscape Environment de Unreal Engine per a Windows i Mac i adaptar-lo per a ser utilitzat amb l’editor UE4Editor d’Ubuntu Linux.

        2.2. Editar l'entorn "Landscape environment" per a adequar-lo al requeriments del projecte

        2.3. Ús de múltiples drons (simulació de l’eixam)

        2.4. Configuració de dispositius accessoris per a tasques de depuració i testeig

        2.5. OPCIONAL: Disseny d’un hexacòpter

    3. FITA#03 – Preparació d'un entorn virtual adequat per a la realització de la PoC 

        3.1. Tasques de recerca i investigació

            3.1.1.	Estudi de la API de AirSim per a Python
	
            3.1.2. 	Recerca de projectes basats en AirSim

        3.2. preparació dels elements de l'entorn UE "LandscapeMountains Environment"

            3.2.1. Simulació d'incendis forestals

            3.2.2. Simulació de condicions nocturnes d'iluminació 

        3.3. Preparació dels actors: Modes de AirSim

            3.3.1. Mode "Computer Vision"

                3.3.1.1. Captura aleatòria d'imatges per a generar els jocs de proves

                3.3.1.2. Simulació de visió tèrmica nocturna tipus FLIR per a generar els training data-set

            3.3.2. Mode "Multirotor"

                3.3.2.1. Implementació de mode “patrulla” Individual (script "drone_patrol.py")

                3.3.2.2. Implementació de mode mode “patrulla” Col·lectiu (script "swarm_patrol.py")

        3.4. Wrapping-up: Generació de l'entorn final per a la realització de la PoC

    4. FITA#04 - Generar el dataset d'imatges per a entrenar i testejar el model de xarxes neuronals profundes

    5. FITA#05 – Disseny de l’arquitectura de xarxes neuronals profundes
 
        5.1. Tasques d’investigació i recerca

        5.2. Estructuració de l'algorisme de DL i configuració d'hyperparàmetres

            5.2.1. Identificació / definició de les mètriques de rendiment

            5.2.2. Establiment / disseny d'un model base (baseline model)

                5.2.2.1. AlexNet
				
				5.2.2.2. DenseNet121

            5.2.3. Preparació de les dades per a l'entrenament del model

                    5.2.3.1. Pre-processament de les imatges del dataset 
					
					5.2.3.2. Divisió dels dataset en grups 

        5.3. Avaluació del model i interpretació del seu rendiment

        5.4. Millores en el rendiment de la xarxa neuronal i reajustament d’hiper-paràmetres 
		
		    5.4.1. Dropout Regularization
			
        5.5. Estudi del rati d’aprenentatge i optimització dels paràmetres
		
        5.6. Wrapping-Up: Adaptació i implementació de les arquitectures DCNN escollides 

--------------------------------------------------------------

6.	FITA#06 - Entrenar la DCNN mitjançant el training data-set apropiadament normalitzat

7.	FITA#07 - Testeig de la DCNN (Joc de proves #1):

7.1. Definir un joc de proves reduït per a testar el correcte funcionament de la DCNN amb un sol dron

7.2. Entrenar i fer el deployment d’un sol dron sobre el joc de proves #1

7.3. Avaluar el comportament de la DCNN: En cas de fallada repassar el punt 4, o bé repetir el punt 5 amb un training data-set amb un conjunt d’imatges més gran. En cas d’èxit seguir amb el punt 7)

8.	FITA#08 - Entrenar la resta de drons de l’eixam (4 unitats de drons o octocòpters)

9.	FITA#09 - Disseny + implementació dels mecanismes de comunicació entre els drons de l’eixam:

9.1. Definir els mecanismes de comunicació que utilitzaran els drons tant pel que fa la rebuda de comandes de control com pel que respecta al pas de missatges a la resta de l’eixam 

9.2. Descriure amb més detall les PoC que prendrem com a referència per a implementar la comunicació: quines ordres es comuniquen, quines dades s’envien (e.g. coordenades GPS) i quines accions s’efectuen

9.3. Definir un sistema de decisió de prioritat dels missatges inter-dron, així com establir com es gestiona el cas en què dos o més missatges amb la mateixa prioritat arribin “a l’hora”. A tall d’exemple; si dos drons envien un mateix missatge a l’hora es determina quin és el de més prioritat i s’actua en conseqüència, però si tenen la mateixa prioritat, aleshores per a determinar quina ordre s’accepta o es descarta s’haurà d’utilitzar un mecanisme de decisió distribuït; o bé un de senzill basat en timestamps en cas que els drons (nodes) de l’eixam estiguin perfectament sincronitzats, o bé, en cas de que no ho estiguin, utilitzar altres mecanismes més complexos com per exemple Lamport timestamp o bé Vector Clocks. 

9.4. Definir quin mecanisme (distribuït) s’utilitza per a propagar missatges per tot l’eixam. p.e. Gossip Protocol, Time-Stamped Anti-Entropy Protocol (TSAE), etc.

9.5. Implementar els mecanismes de comunicació distribuïda que s’hagi decidit utilitzar

10.	FITA#10 - Testeig de les comunicacions (joc de proves #2): 

10.1. Definir un joc de proves per tal de comprovar que la comunicació inter-dron s’efectua de manera correcta

10.2. Executar el joc de proves sobre tot l’eixam

10.3. Avaluar els resultats del testeig. Si falla tornar al punt 8, si no, passar al punt 10.

11.	FITA#11 - Testeig de tot el sistema (joc de proves #3):

11.1. Unir i estendre els jocs de proves #1 i #2 per a poder provar els casos d’ús (PoC) 
definits  a l’abstract del projecte.  

11.2. Fer el Deployment de tot l’eixam sobre el joc de proves nº 3

11.3. Avaluar resultats i procedir en conseqüència

12.	FITA#12 - Obtenir conclusions i redactar la memòria del projecte

13.	FITA#13 - Crear una presentació i/o vídeo-demo del projecte que destaqui els punts principals i demostri el correcte funcionament de tot el sistema sobre les PoC definides.

14.	FITA#14 - Preparar la defensa del projecte


## 4 - Bibliografia

Salvatierra, Issac; Bosch, Francina; Marfà, Ricard; Longan, Idoia. Per què Catalunya crema [Internet]. Barcelona: [Actualitzat el (N/A) ; consultat l’1 d’octubre de 2021]. Disponible a:  https://interactius.ara.cat/incendis/

Elgendy, Mohamed. Deep Learning for Vision Systems. Nova York, Estats Units d’Amèrica: Manning publications; 2020. 458 p.

Yadav, Robin. Deep Learning Based Fire Recognition for Wildfire Drone Automation. The Canadian Science Fair Journal. 2020 Oct;3(2):N/A. https://csfjournal.com/volume-3-issue-2/2020/10/30/deep-learning-based-fire-recognition-for-wildfire-drone-automation

Microsoft Research. Aerial Informatics and Robotics Platform [Internet]. Microsoft: Microsoft Research; 2017 [Actualitzat el 5 de febrer de 2021; consultat el l’9 d’octubre de 2021]. https://www.microsoft.com/en-us/research/project/aerial-informatics-robotics-platform/#overview

Universitat Oberta de Catalunya. Com citar: Estil Vancouver [Internet]. Campus virtual: Recursos d’aprenentatge de l’assignatura “TFG – Arquitectura de computadors i sistemes operatius”; [Actualitzat el (N/A); consultat el 9 d’octubre de 2021]. Disponible a:  https://biblioteca.uoc.edu:8080/ca/plana/Estil-Vancouver/ 

Beneito Montagut, Roser. Presentació de documents i elaboració de presentacions [Internet]. Barcelona: Editorial UOC; data de publicació no disponible [consultat el 9 d’octubre de 2021]. 56 p. Disponible a: https://campus.uoc.edu/cdocent/HKZB5UG6XS130_6R5O43.pdf 
