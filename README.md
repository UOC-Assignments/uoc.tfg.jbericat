# Universitat Oberta de Catalunya - Treball Final de Grau 
# *Arquitectura de computadors i sistemes operatius*

Estudiant: Jordi Bericat Ruz

Professor col·laborador: Daniel Rivas Barragan

Semestre: Tardor 2021/22 (Aula 1) 

## Índex

1. [Títol del projecte](#1---t%C3%ADtol-provisional-del-projecte)

2. [Resum del projecte: Motivació i definició dels objectius](#2---resum-provisional-del-projecte-motivaci%C3%B3-i-definici%C3%B3-dels-objectius)

3. [Definició dels objectius del TFG: Planificació temporal](#3---definici%C3%B3-dels-objectius-del-tfg-planificaci%C3%B3-temporal)

    3.1. [Planificació temporal per fites (Google Calendar)](#31--planificaci%C3%B3-temporal-per-fites-google-calendar)

    3.2. [Descripció de cada fita i detall de tasques relacionades](#32--descripci%C3%B3-de-cada-fita-i-detall-de-tasques-relacionades)

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

    1. FITA#01 – Establir i preparar l’entorn de desenvolupament
 
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

    2. FITA#02 – Adaptació i personalització de l’entorn de la plataforma de simulació de vehicles autònoms AirSim de Microsoft

        2.1. Obtenir l’entorn Landscape Environment de Unreal Engine per a Windows i Mac i adaptar-lo per a ser utilitzat amb l’editor UE4Editor d’Ubuntu Linux.

        2.2. Editar l'entorn "Landscape environment" per a adequar-lo al requeriments del projecte

        2.3. Ús de múltiples drons (simulació de l’eixam)

        2.4. Configuració de dispositius accessoris per a tasques de depuració i testeig

        2.5. OPCIONAL: Disseny d’un hexacòpter

    3. FITA#03 – Preparació d'un entorn virtual adequat per a la realització de la PoC 

        3.1. Tasques de recerca i investigació

        3.2. Modificacions de l’entorn base de Unreal: “LandscapeMountains Environment"

                3.2.1. Simulació d'incendis forestals

                        3.2.1.1. Classe #1: Incendis d’alta intensitat

                        3.2.1.2. Classe #2: Incendis d’intensitat moderada

                        3.2.1.3. Classe #3: Incendis de baixa intensitat

                3.2.2. Simulació de condicions nocturnes d’il·luminació

        3.3. Wrapping-up: Generació de l’entorn final per a la realització de la PoC

    4. FITA#04 – Creació del dataset d'imatges per a entrenar i testejar el model de xarxes neuronals profundes

        4.1. Tasques de recerca i investigació
		
            4.1.1. Estructura del dataset

            4.1.2. Estudi de la API d’AirSim per a Python

            4.1.3. Recerca de projectes basats en AirSim
			
        4.2. Establiment dels objectius de classificació
		
        4.3. Definició de l’estructura d’etiquetat del classificador
		
            4.3.1. Elaboració de l’estructura d’etiquetat

            4.3.2. Representació de l’etiquetat: Estructura de directoris i noms d’arxiu

            4.3.3. Quantitat de mostres necessàries per a l’entrenament i testeig del model
		
        4.4. Assignació de les zones definides a l’entorn per a l’obtenció d’imatges
		
        4.5. Configuració dels paràmetres de captura de les imatges

        4.6. Modificació de l'script Python utilitzar per a la captura d'imatges
	
            4.6.1. Simulació de visió tèrmica nocturna tipus FLIR durant la obtenció d’imatges del dataset

            4.6.2. Automatització de la presa d’imatges i del seu etiquetat (labeling)

        4.7. Recol·lecció i etiquetat d'imatges (execució de l’script de captura d’imatges capture_ir_segment.py)
		
        4.8. Wrapping-up: Empaquetament i publicació del dataset d’imatges per a la realització de la PdC
			
    5. FITA#05 – Disseny de l’arquitectura de xarxes neuronals profundes (CNN)
 
        5.1. Tasques d’investigació i recerca

                5.1.1. Repàs cronològic dels models de Deep Learning més destacats i estudi comparatiu de les seves característiques

                        5.1.1.1. AlexNet
        
                        5.1.1.2. DenseNet121

        5.2. Estructuració de l'algorisme de Deep Learning

                5.2.1. Preparació de les dades per a l'entrenament del model

                        5.2.1.1. Pre-processament de les imatges del dataset

                        5.2.1.2. Divisió dels dataset en grups

                5.2.2. Definició d’un model base (baseline model)
                        
                        5.2.2.1. Versió 1.0

                        5.2.2.2. Versió 2.0

                        5.2.2.3. Versió 3.0

                5.2.3. Càlcul d'hiper-paràmetres
                        
                        5.2.3.1. Versió 1.0

                        5.2.3.2. Versió 2.0

                        5.2.3.3. Versió 3.0

                5.2.4 – Implementació de l’estructura del model CNN amb el framework pyTorch
                        
                        5.2.4.1. Versió 1.0

                        5.2.4.2. Versió 2.0

                        5.2.4.3. Versió 3.0

                5.2.5 – Definició de la funció de pèrdua (loss function)

        5.3. Wrapping-Up: Implementació d'un algorisme d'entrenament i generació de gràfiques i estadístiques

    6. FITA#06 – Entrenament de la CNN

        6.1. Tasques d’investigació i recerca

                6.1.1. Definició de les mètriques de rendiment

        6.2. Selecció del model i del dataset finals que s'utilitzaran per a la PdC
		
                6.2.1. Selecció del model

                6.2.2. Selecció del dataset
				
                        6.2.2.1. Dataset v4.0
						
                        6.2.2.2. Dataset v5.0
						
                        6.2.2.3. Dataset v6.0
						
                        6.2.2.4. Dataset v7.0-beta
						
                        6.2.2.5. Dataset v7.0
						
                        6.2.2.6. Dataset v8.0

                        6.2.2.6. Dataset v9.0

        6.3. Entrenament del model amb el dataset seleccionat

        6.4. Avaluació del model i interpretació del seu rendiment (estudi del rati d’aprenentatge)

        6.5. Millores en el rendiment (optimització de paràmetres)
		
	7. FITA#07 – Execució de la PdC: Explotació del model CNN

		7.1. Tasques de recerca i investigació
				
		7.2. Preparacions finals per a la execució de la PdC
		
				7.2.1. Adaptació de l'entorn UE4 "LME" per a la execució de la PdC

				7.2.2. Ajustos del fitxer de configuració d’AirSim
				
                7.2.3. Re-estructuració dels arxius font del repositori del projecte

		7.3. Implementacions necessàries per a la execució de la PdC
		
				7.3.1. Optimització de la funció create_flir_img() per a la simulació d'imatges FLIR
				
				       7.3.1.1. Versió 1: Algorisme sense optimitzar
						
                       7.3.1.2. Versió 2: Algorisme optimitzat

				7.3.2. Implementació de la funció de simulació d'imatges FLIR "flir_offline_batch_converter.py"
				
                7.3.3. Implementació de l’algorisme d'inferència del model CNN: "cnn_deployment.py
						
				7.3.4. Implementació de la identificació de les classes detectades pel model mitjançant bounding boxes: funció “add_bounding_boxes()
				
                7.3.5. Creació d'scripts bash per a l'execució de codi Python

		7.4. Execució de la PdC
		
		7.5. ANNEX: Control dels canvis efectuats als fitxers de codi font durant la darrera etapa del projecte
		
	8. FITA#08 –  Prova de concepte: Optimitzacions, conclusions i consideracions finals

		8.1. Primera execució de la prova de concepte
		
				8.1.1. Conclusions obtingudes després de la primera execució de la prova
				
						8.1.1.1. Problemes de disseny
						
						8.1.1.2. Problemes amb el dataset i les característiques de les imatges
						
						8.1.1.3. Verificació del Model CNN entrenat

						8.1.1.4. Problemes relacionats amb la inferència del model
				
				8.1.2. recol·lecció de feedback i implementació de millores
				
						8.1.2.1. Dataset d’imatges
						
						8.1.2.2. Algorisme d’entrenament
						
						8.1.2.3. Algorisme d’inferència
						
		8.2. Segona execució de la prova de concepte
		
				8.2.1. Preparació de la prova
				
						8.2.1.1. Entrenament del model amb el nou dataset
						
				8.2.2. Inferència del model sobre el mateix conjunt d’imatges d’explotació utilitzat a la primera PdC
				
				8.2.3. Conclusions obtingudes després de la segona execució de la prova
				
				8.2.4. Recol·lecció de feedback i implementació de millores
				
						8.2.4.1. Dataset d’imatges
						
						8.2.4.2. Algorisme d’entrenament
						
						8.2.4.3. Algorisme d’inferència
						
		8.3. Tercera execució de la prova de concepte
		
				8.3.1. Preparació de la prova
		
						8.3.1.1. Entrenament del model amb l’algorisme revisat a la secció 8.2.2.2
						
				8.3.2. Inferència de les dades
				
				8.3.3. Conclusions obtingudes després de la tercera execució de la prova
						
		8.4. Conclusions finals				
						
		8.5. Consideracions addicionals al respecte de l’abast, objectius i compromisos establerts durant la planificació inicial d’aquest projecte
						
--------------------------------------------------------------


## 4 - Bibliografia

Salvatierra, Issac; Bosch, Francina; Marfà, Ricard; Longan, Idoia. Per què Catalunya crema [Internet]. Barcelona: [Actualitzat el (N/A) ; consultat l’1 d’octubre de 2021]. Disponible a:  https://interactius.ara.cat/incendis/

Elgendy, Mohamed. Deep Learning for Vision Systems. Nova York, Estats Units d’Amèrica: Manning publications; 2020. 458 p.

Yadav, Robin. Deep Learning Based Fire Recognition for Wildfire Drone Automation. The Canadian Science Fair Journal. 2020 Oct;3(2):N/A. https://csfjournal.com/volume-3-issue-2/2020/10/30/deep-learning-based-fire-recognition-for-wildfire-drone-automation

Microsoft Research. Aerial Informatics and Robotics Platform [Internet]. Microsoft: Microsoft Research; 2017 [Actualitzat el 5 de febrer de 2021; consultat el l’9 d’octubre de 2021]. https://www.microsoft.com/en-us/research/project/aerial-informatics-robotics-platform/#overview

Universitat Oberta de Catalunya. Com citar: Estil Vancouver [Internet]. Campus virtual: Recursos d’aprenentatge de l’assignatura “TFG – Arquitectura de computadors i sistemes operatius”; [Actualitzat el (N/A); consultat el 9 d’octubre de 2021]. Disponible a:  https://biblioteca.uoc.edu:8080/ca/plana/Estil-Vancouver/ 

Beneito Montagut, Roser. Presentació de documents i elaboració de presentacions [Internet]. Barcelona: Editorial UOC; data de publicació no disponible [consultat el 9 d’octubre de 2021]. 56 p. Disponible a: https://campus.uoc.edu/cdocent/HKZB5UG6XS130_6R5O43.pdf 
