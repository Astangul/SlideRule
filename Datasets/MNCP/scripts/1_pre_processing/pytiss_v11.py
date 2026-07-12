#######################################################################
# IMPORTATION DES BIBLIOTHEQUES
#######################################################################

from xml.dom import minidom
from xml.dom.minidom import Document
from collections import deque
import sys, os, re, subprocess, copy
from math import *

#######################################################################
# DEFINITION DES CLASSES
#######################################################################

#######################################################################
class Variable:

    def __init__(self, nom, chaine):
        self.nom = nom
        self.chaine = chaine
        self.tableau = chaine.split()

#######################################################################
class Groupement:

    def __init__(self, nom):
        self.nom = nom
        self.listeVariables = deque()
        self.lenMax = 0
        self.lenMin = 1000000

#######################################################################
class Dossier:

    def __init__(self):
        self.nomFichier = ""
        self.cheminFichier = ""

#######################################################################
class ListeDesVariables:

    def __init__(self, listeVariablesClass, listeGroupementsClass, listeFormulesClass):
        # liste des variables
        self.listeVariables = listeVariablesClass
        
        # liste des formules
        self.listeFormules = listeFormulesClass
        
        # liste des groupements
        self.listeGroupements = listeGroupementsClass

        # liste des tableaux d'Uplet
        self.listeUplet = deque()

        # Tableau compose du nombre de variable ou
        # du nombre max de variable pour chaque groupement
        self.tableauLimites = deque()
        for variable in listeVariablesClass:
            self.tableauLimites.append(len(variable.tableau))
        for groupement in listeGroupementsClass:
            self.tableauLimites.append(groupement.lenMax)

        # Tableau permettant de stocker la position de l iteration
        # pour generer les fichiers
        self.tableauParam = [0] * len(self.tableauLimites)

        # Tableau stockant les variables
        self.tableauUplet = deque()

        # Construction de deux listes afin de stocker les variables
        # calculables et les chaines
        self.variablesCalculables = deque()
        self.chaines = deque()
        self.allVariables = deque()

        # Indicateur de doublon
        self.isVariablePresenteXfois = deque()

    # verification de la validite de tableau param
    def check(self):
        # Debut de l iteration au bout du tableau
        positionTableauParam = len(self.tableauParam) - 1

        # Iteration sur la position dans le tableau param
        # Arret de l iteration a la deuxieme valeur du tableau
        # => ajouter une condition d arret lorsque la premiere
        # colonne depasse le nombre de parametre
        while positionTableauParam > 0:
            # si la valeur dans le tableau param depasse le nombre de
            # parametre max dans la
            # variable ou le groupement on met la valeur du tableau a 0
            # et on ajoute 1 a la position inferieure
            if self.tableauLimites[positionTableauParam] <= \
            self.tableauParam[positionTableauParam]:
                self.tableauParam[positionTableauParam] = 0
                self.tableauParam[positionTableauParam - 1] += 1
            # Descente du tableau
            positionTableauParam -= 1

    # chargement des parametre dans tableau uplet
    def chargementUplet(self):
        # RaZ
        self.tableauUplet.clear()

        # Iteration sur les variables
        compteur = 0
        while compteur < len(self.listeVariables):
            self.tableauUplet.append([self.listeVariables[compteur].nom,
            self.listeVariables[compteur].tableau[self.tableauParam[compteur]]])
            compteur += 1

        # Iterations sur les groupements
        while compteur < len(self.tableauLimites):
            # chargement du groupement
            groupement = self.listeGroupements[compteur -
            len(self.listeVariables)]

            # liste des variable du groupzmznt pour ajout au tableau uplet
            listeVariablesGroupement = deque()
            numVariable = self.tableauParam[compteur]
            for variable in groupement.listeVariables:
                if numVariable < len(variable.tableau):
                    listeVariablesGroupement.append([variable.nom,
                    variable.tableau[numVariable]])
                else:
                    listeVariablesGroupement.append([variable.nom,
                    variable.tableau[len(variable.tableau) - 1]])

            # Ajout de la liste au tableau de uplet
            self.tableauUplet.append(listeVariablesGroupement)
            compteur += 1

    def constructionListesVariablesCalculablesOuNon(self):
        # Afin de discriminer si dans un tableauUplet
        # on a affaire a une liste (=> groupement)
        # ou a un uplet (=> variable)
        # on fait :
        #  if type(i) == type(list()):
        #     #do something with a list
        # elif type(i) == type(tuple()):
        #     #do something with a tuple
        # elif type(i) == type(str()):
        #     #here's your string

        # RaZ
        # Construction de deux listes afin de stocker les variables
        # calculables et les chaines
        self.variablesCalculables = deque()
        self.chaines = deque()
        self.allVariables = deque()

        # Verification presence variable
        def rechercheVar(self, variable):
            for tableau in self.allVariables:
                if tableau[0] == variable:
                    self.isVariablePresenteXfois.append(variable)
            pass

        # Discrimination des variables calculables
        def ajoutTableau(self, tabList):
            try:
                valeur = float(tabList[1])
                self.variablesCalculables\
                .append([tabList[0], valeur])

                rechercheVar(self, tabList[0])
                self.allVariables\
                .append([tabList[0], "%.10" % valeur])
            except Exception:
                self.chaines\
                .append([tabList[0], tabList[1]])

                rechercheVar(self, tabList[0])
                self.allVariables\
                .append([tabList[0], tabList[1]])

        # Boucle sur le tableau d uplet
        for tabList in self.tableauUplet:
            if type(tabList) == type(list()):
                ajoutTableau(self, tabList)
            elif type(tabList) == type(deque()):
                for tableau in tabList:
                    ajoutTableau(self, tableau)
            else:
                print("erreur avec ")
                print(tabList)
                
        # Tri des listes afin d eviter les erreurs de remplacement
        self.variablesCalculables = \
        sorted(self.variablesCalculables,
        key=lambda item: -len(item[0]))

        self.chaines = \
        sorted(self.chaines, key=lambda item: -len(item[0]))

        self.allVariables = \
        sorted(self.allVariables,
        key=lambda item: -len(item[0]))
        
        # Boucle sur les variables afin de remplacer les formules du XML
        for variableClass in self.listeFormules:
            formule = variableClass.chaine
            # Remplacement des variables dans la formule
            for tab in self.variablesCalculables:
                formule = formule.replace(tab[0], "%.10e" % tab[1])
            try:
                valeur = eval(formule)
                self.variablesCalculables\
                .append([variableClass.nom, valeur])

                rechercheVar(self, variableClass.nom)
                self.allVariables\
                .append([variableClass.nom, "%.10e" % valeur])
            except Exception:
                self.chaines\
                .append([variableClass.nom, formule])

                rechercheVar(self, variableClass.nom)
                self.allVariables\
                .append([variableClass.nom, formule])
			
			# Tri des listes afin d eviter les erreurs de remplacement
            self.variablesCalculables = \
            sorted(self.variablesCalculables,
            key=lambda item: -len(item[0]))
            self.chaines = \
            sorted(self.chaines, key=lambda item: -len(item[0]))
            self.allVariables = \
            sorted(self.allVariables,
            key=lambda item: -len(item[0]))

#######################################################################
# CLASSE Traitement du texte
#######################################################################
class TraitementTexte:

    def __init__(self, nom_fichier):
        ## -------------------------------------
        # Declaration des variables de la classe
        ## -------------------------------------
        self.nomGeneriqueFichier = ""
        self.scriptLancement = ""
        self.listeNomDeFichier = deque()
        self.listeVariablesClass = deque()
        self.listeFormulesClass = deque()
        self.listeGroupementsClass = deque()
        self.texteXML = ""
        self.isXMLfind = False
        self.ouvertureAccolades = deque()
        self.fermetureAccolades = deque()

        self.ouvertureAccoladesNom = deque()
        self.fermetureAccoladesNom = deque()

        self.texteFichier = ""
        self.listeFormulesInitial = deque()
        self.listeFormulesModifiees = deque()

        self.listeFormulesNomInitial = deque()
        self.listeFormulesNomModifiees = deque()

        self.compteurGenerationFichier = 0
        
        self.docOutXML = Document()
        self.rootXML = self.docOutXML.createElement('Resume')
        self.docOutXML.appendChild(self.rootXML)
        
        self.listeBlocsTexte = {} # Ajouter cet attribut pour stocker les blocs de texte

        # Ouverture du fichier de donnees sous forme d un tableau
        fichier = open(nom_fichier, 'r')
        tableauTexteFichier = fichier.readlines()

        # Ecriture du fichier sous la forme d une chaine
        texteFichier = ""
        for texte in tableauTexteFichier:
                texteFichier += texte

        # Recuperation de la partie XML du fichier
        texteXML = re.compile('<donnees>(.*?)</donnees>', re.DOTALL).\
        findall(texteFichier)

        # Transformation de/des partie(s) recuperee(s) en texte
        # Declaration d une variable pour la class
        # Conservation du texteXML
        self.texteXML = ""
        for texte in texteXML:
                self.texteXML += texte
        self.texteXML = "<donnees>\r" + self.texteXML + "</donnees>"

        # Conservation des lignes apres la ligne </donnees>
        isDonneesFind = False
        for texte in tableauTexteFichier:
                if isDonneesFind:
                        self.texteFichier += texte
                else:
                        if re.search('</donnees>', texte):
                                isDonneesFind = True

    #######################################################################
    # RECUPERATION DES GROUPEMENTS ET DES VARIABLES
    #######################################################################
    def recupVarGroupementDuXML(self):
        try:
            # Creation du minidom
            xmldoc = minidom.parseString(self.texteXML)

            # Creation du childNodes
            cNodes = xmldoc.childNodes

            # Exploration de l XML
            self.isXMLfind = xmldoc.hasChildNodes()
            if self.isXMLfind:
                if cNodes.length == 1 and cNodes[0].nodeName == "donnees":
                    #--------------------------------------------------------
                    # Travail sur les variables
                    # -------------------------------------------------------

                    # Recuperation du nom du fichier
                    nomFichierListe = cNodes[0].getElementsByTagName("nom")
                    for nomFich in nomFichierListe:
                        if nomFich.parentNode.nodeName == "donnees":
                            self.nomGeneriqueFichier = \
                            nomFich.childNodes[0].nodeValue
                            self.nomGeneriqueFichier = \
                            self.nomGeneriqueFichier.replace("\t","")
                            break

                    # Recuperation du nom du fichier
                    nomFichierListe = cNodes[0].getElementsByTagName("script")
                    for nomFich in nomFichierListe:
                        if nomFich.parentNode.nodeName == "donnees":
                            self.scriptLancement =\
                            nomFich.childNodes[0].nodeValue
                            self.scriptLancement = \
                            self.scriptLancement.replace(" ", "")
                            break

                    # Creation de la node list : variable
                    variableList = cNodes[0].getElementsByTagName("variable")

                    # Exploration de la liste des variables
                    # On boucle sur les variables
                    for variable in variableList:
                        if variable.parentNode.nodeName == "donnees":
                            variableName = variable.getAttribute("nom")
                            variableValue = variable.childNodes[0].nodeValue
                            variableClass = Variable(variableName, variableValue)
                            if len(variableClass.tableau) > 0:
                                self.listeVariablesClass.append(variableClass)
                        elif variable.parentNode.nodeName == "formules":
                            variableName = variable.getAttribute("nom")
                            variableValue = variable.childNodes[0].nodeValue
                            variableClass = Variable(variableName, variableValue)
                            if len(variableClass.tableau) > 0:
                                self.listeFormulesClass.append(variableClass)

                    # -------------------------------------------------------
                    # Travail sur les groupements
                    # -------------------------------------------------------

                    # Creation de la node list : groupement
                    groupementList = cNodes[0].getElementsByTagName("groupement")

                    # On boucle sur les differents groupements
                    for groupement in groupementList:

                        # Nom du groupement
                        groupementName = groupement.getAttribute("nom")

                        # Creation d une class Groupement
                        groupementClass = Groupement(groupementName)

                        # On boucle sur les variables du groupement
                        for variableGroupement in groupement.\
                        getElementsByTagName("variable"):
                            variableName = variableGroupement.getAttribute("nom")
                            variableValue = variableGroupement.childNodes[0].\
                            nodeValue
                            variableClass = Variable(variableName, variableValue)
                            if len(variableClass.tableau) > 0:
                                groupementClass.listeVariables.append(variableClass)
                                groupementClass.lenMax = max(groupementClass.lenMax,
                                len(variableClass.tableau))
                                groupementClass.lenMin = min(groupementClass.lenMin,
                                len(variableClass.tableau))

                        # ajout du groupement a la liste de groupement
                        self.listeGroupementsClass.append(groupementClass)
        except Exception:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("Revoir le formatage de votre XML")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            pass
        

    #--------------------------------------------------------------
    # Boucle sur le fichier texte afin de reperer les {}
    # Si incoherence dans les {}
    #--------------------------------------------------------------
    def recuperationAccolades(self):
        # Fonction permettant de recuperer les accollades dans un fichier texte
        def recupGeneAccolades(self, texteFichier):
            # RaZ
            ouvertureAccolades = deque()
            fermetureAccolades = deque()

            # Boucle sur le fichier
            compteurFichier = 0
            while compteurFichier < len(texteFichier):
                # Si on trouve une accolade ouverte
                if texteFichier[compteurFichier] == "{":
                    ouvertureAccolades.append(compteurFichier)

                # Si on trouve une accolade fermee
                elif texteFichier[compteurFichier] == "}":

                    lenOuvertureAccolade = len(ouvertureAccolades)
                    lenFermetureAccolade = \
                    len(fermetureAccolades) + 1

                    # Verification de la concordance des listes
                    if lenOuvertureAccolade == lenFermetureAccolade:

                        # Ajout a la liste d accolade
                        fermetureAccolades.append(compteurFichier)
                        if ouvertureAccolades[
                            lenOuvertureAccolade - 1] \
                        >= fermetureAccolades[
                            lenFermetureAccolade - 1]:
                            fermetureAccolades.append(1)
                            return ouvertureAccolades, fermetureAccolades
                    else:
                        if lenOuvertureAccolade == lenFermetureAccolade - 1:
                            fermetureAccolades.append(1)
                        return ouvertureAccolades, fermetureAccolades
                # Incrementation de la boucle sur le fichier Texte
                compteurFichier += 1

            # Retour des listes
            return ouvertureAccolades, fermetureAccolades

        # Recup des accolades
        #++++++++++++++++++++
        self.ouvertureAccolades, self.fermetureAccolades \
        = recupGeneAccolades(self, self.texteFichier)

        self.ouvertureAccoladesNom, self.fermetureAccoladesNom \
        = recupGeneAccolades(self, self.nomGeneriqueFichier)

    #------------------------------------------
    # Recuperation des formules au format texte
    #------------------------------------------
    def recuperationFormules(self):
        # Boucle sur la liste des accolades
        # Afin d ajouter dans une liste toutes les formules

        # RecuperationFormulesGeneriques
        def recupGeneFormules(ouvertureAccolades, fermetureAccolades,
        texteFichier):
            # Raz
            listeFormulesInitial = deque()

            # Boucle sur les accolades
            compteurAccolade = 0
            while compteurAccolade < len(ouvertureAccolades):
                nTextAccolInf = ouvertureAccolades[compteurAccolade] \
                + 1
                nTextAccolSup = fermetureAccolades[compteurAccolade] \
                - 1
                if nTextAccolSup > nTextAccolInf:
                    formule = ""
                    for posiTextFichier in range(nTextAccolInf,
                    nTextAccolSup + 1):
                        formule += texteFichier[posiTextFichier]
                    if len(formule.split()) > 0:
                        formule = formule.replace('(',' ( ')
                        formule = formule.replace(')',' ) ')
                        formule = formule.replace('+',' + ')
                        formule = formule.replace('-',' - ')
                        formule = formule.replace('*',' * ')                
                        formule = formule.replace('/',' * 1.0 / ')
                        formule = formule.replace('*  *',' ** ')
                        formule = formule.replace('E + ','E+')
                        formule = formule.replace('E - ','E-')
                        formule = formule.replace('e + ','e+')
                        formule = formule.replace('e - ','e-')
                        listeFormulesInitial.append(formule)
                    else:
                        listeFormulesInitial.append("")
                elif nTextAccolSup == nTextAccolInf:
                    formule = ""
                    formule = texteFichier[nTextAccolSup]
                    if len(formule.split()) > 0:
                        formule = formule.replace('(',' ( ')
                        formule = formule.replace(')',' ) ')
                        formule = formule.replace('+',' + ')
                        formule = formule.replace('-',' - ')
                        formule = formule.replace('*',' * ')                
                        formule = formule.replace('/',' * 1.0 / ')
                        formule = formule.replace('*  *',' ** ')
                        formule = formule.replace('E + ','E+')
                        formule = formule.replace('E - ','E-')
                        formule = formule.replace('e + ','e+')
                        formule = formule.replace('e - ','e-')
                        listeFormulesInitial.append(formule)
                else:
                    listeFormulesInitial.append("")
                compteurAccolade += 1

            # Fin de la boucle
            return listeFormulesInitial

        # Calcul de la liste de formules
        self.listeFormulesInitial = recupGeneFormules(self.ouvertureAccolades,
        self.fermetureAccolades, self.texteFichier)

        self.listeFormulesNomInitial = recupGeneFormules(
        self.ouvertureAccoladesNom, self.fermetureAccoladesNom,
        self.nomGeneriqueFichier)

    #------------------------------------------
    # Remplacement des variables des formules
    #------------------------------------------
    def remplacementVariablesFormules(self, listeDesVariables):

        # REmplacement Generiques
        def remplaGeneVarFormules(listeFormulesInitial):
            listeFormulesModifiees = deque()
            for formule in listeFormulesInitial:
                formule = formule.replace("\t", " ")
                splitFormule = formule.split(";")
                if (len(splitFormule) == 2):
                    splitFormule[0] = splitFormule[0].replace(" ","")
                    try:
                        formuleBis = splitFormule[1]
                        tableauFormuleBis = formuleBis.split(" ")
                        formuleBis = ""
                        for texte in tableauFormuleBis:
                            # Remplacement des parametres par les valeurs
                            # dans une formules
                            for tableau in listeDesVariables.variablesCalculables:
                                if texte == str(tableau[0]):
                                    texte = "%.10e" % tableau[1]
                                elif re.search(r"^[0-9]", texte) is None:
                                    texte = texte.replace("+", " + ")
                                    texte = texte.replace("-", " - ")
                                    tab = texte.split(" ")
                                    texte = ""
                                    for partie in tab:
                                        if partie == str(tableau[0]):
                                            partie = "%.10e" % tableau[1]
                                        texte += partie + " "
                            formuleBis += texte + " "
                        formuleBis = ("%." + splitFormule[0]) % eval(formuleBis)
                        listeFormulesModifiees.append(formuleBis)
                    except Exception:
                        tableauFormuleBis = splitFormule[1].split(" ")
                        splitFormule[1] = ""
                        for texte in tableauFormuleBis:
                            # Remplacement des parametres par les valeurs
                            # dans une formules
                            for tableau in listeDesVariables.allVariables:
                                if texte == str(tableau[0]):
                                    texte = str(tableau[1])
                                elif re.search(r"^[0-9]", texte) is None:
                                    texte = texte.replace("+", " + ")
                                    texte = texte.replace("-", " - ")
                                    tab = texte.split(" ")
                                    texte = ""
                                    for partie in tab:
                                        if partie == str(tableau[0]):
                                            partie = str(tableau[1])
                                        texte += partie + " "
                            splitFormule[1] += texte + " "
                        listeFormulesModifiees.append(splitFormule[1])
                        pass
                else:
                    try:
                        formuleBis = formule
                        tableauFormuleBis = formuleBis.split(" ")
                        formuleBis = ""
                        for texte in tableauFormuleBis:
                            # Remplacement des parametres par les valeurs
                            # dans une formules
                            for tableau in listeDesVariables.variablesCalculables:
                                if texte == str(tableau[0]):
                                    texte = "%.10e" % tableau[1]
                                elif re.search(r"^[0-9]", texte) is None:
                                    texte = texte.replace("+", " + ")
                                    texte = texte.replace("-", " - ")
                                    tab = texte.split(" ")
                                    texte = ""
                                    for partie in tab:
                                        if partie == str(tableau[0]):
                                            partie = "%.10e" % tableau[1]
                                        texte += partie + " "
                            formuleBis += texte + " "
                        formuleBis = eval(formuleBis)
                        listeFormulesModifiees.append(formuleBis)                        
                    except Exception:
                        tableauFormuleBis = formule.split(" ")
                        formule = ""
                        for texte in tableauFormuleBis:
                            # Remplacement des parametres par les valeurs
                            # dans une formules
                            for tableau in listeDesVariables.allVariables:
                                if texte == str(tableau[0]):
                                    texte = str(tableau[1])
                                elif re.search(r"^[0-9]", texte) is None:
                                    texte = texte.replace("+", " + ")
                                    texte = texte.replace("-", " - ")
                                    tab = texte.split(" ")
                                    texte = ""
                                    for partie in tab:
                                        if partie == str(tableau[0]):
                                            partie = str(tableau[1])
                                        texte += partie + " "
                            formule += texte + " "
                        listeFormulesModifiees.append(formule)
                        pass
            return listeFormulesModifiees

        # ajout des formules modifiees
        self.listeFormulesModifiees = remplaGeneVarFormules(
            self.listeFormulesInitial)
        self.listeFormulesNomModifiees = remplaGeneVarFormules(
            self.listeFormulesNomInitial)

    # Ecriture du nouveau texte a partir des formules modifiees
    #----------------------------------------------------------
    def ecritureNouveauTexte(self, texteFichier):

        def ecritureGeneNouveauTexte(texteFichier, ouvertureAccolades, fermetureAccolades, listeFormulesModifiees):
            # RaZ
            nouveauTexte = ""

            # Si aucune accolade n'est présente, renvoyer le texte original
            if not ouvertureAccolades or not fermetureAccolades:
                return texteFichier

            # Première boucle pour le début du texte jusqu'à la première accolade
            nouveauTexte += texteFichier[:ouvertureAccolades[0]]

            # Boucle sur les accolades
            for i in range(len(ouvertureAccolades)):
                # Ajout du remplacement pour l'accolade actuelle
                nouveauTexte += str(listeFormulesModifiees[i])

                # Si ce n'est pas la dernière accolade, ajoutez le texte entre l'accolade fermée actuelle et l'accolade ouverte suivante
                if i < len(ouvertureAccolades) - 1:
                    nouveauTexte += texteFichier[fermetureAccolades[i] + 1:ouvertureAccolades[i + 1]]

            # Ajout du texte après la dernière accolade fermée
            nouveauTexte += texteFichier[fermetureAccolades[-1] + 1:]

            # Renvoi du texte
            return nouveauTexte

        # Modification du Fichier et du nom de fichier
        # Texte du fichier generique
        nouveauTexteFichier = ecritureGeneNouveauTexte(texteFichier, 
        self.ouvertureAccolades, self.fermetureAccolades,
        self.listeFormulesModifiees)
        # Nom du fichier
        if re.search('{', self.nomGeneriqueFichier):
            nouveauTexteNom = ecritureGeneNouveauTexte(self.nomGeneriqueFichier,
            self.ouvertureAccoladesNom, self.fermetureAccoladesNom,
            self.listeFormulesNomModifiees)
            # Remplacement des espaces
            nouveauTexteNom = nouveauTexteNom.replace(" ", "")
        else:
            nouveauTexteNom = self.nomGeneriqueFichier
            # Si le nom du fichier est le meme que le nom generique
            # on numerote
            if (self.nomGeneriqueFichier == nouveauTexteNom):
                nouveauTexteNom += str(self.compteurGenerationFichier)
            # Remplacement des espaces
            nouveauTexteNom = nouveauTexteNom.replace(" ", "")

        # Renvoi des valeurs
        self.listeNomDeFichier.append(nouveauTexteNom)

        # Mis en forme du tableau d Uplet sous forme de liste
        # Utile pour faire les dossiers CRIS
        nouvelleListeUplet = deque()
        for tabList in listeDesVariables.tableauUplet:
            if type(tabList) == type(list()):
                nouvelleListeUplet.append(tabList)
            elif type(tabList) == type(deque()):
                for tableau in tabList:
                    nouvelleListeUplet.append(tableau)
        listeDesVariables.listeUplet.append(nouvelleListeUplet)

        # Renvoi des noms de fichiers
        return nouveauTexteFichier, nouveauTexteNom

    # Ecriture des parametres dans le fichier XML
    #----------------------------------------------------------
    def ecritureSortieXML(self, listeDesVariables):
        
            # Ecriture du nom du fichier
            fichierXML = self.docOutXML.createElement('fichier')
            fichierXML.setAttribute('nom', self.listeNomDeFichier[len(self.listeNomDeFichier) - 1])
            self.rootXML.appendChild(fichierXML)
            
            # Boucle sur les variables
            for tableau in listeDesVariables.allVariables:
                # Creation du noeud variable
                variableXML = self.docOutXML.createElement('variable')
                # Ajout de l attribut nom
                variableXML.setAttribute('nom', tableau[0])
                # Ajout du texte
                text = self.docOutXML.createTextNode(str(tableau[1]))
                variableXML.appendChild(text)
                fichierXML.appendChild(variableXML)
                
    # -------------------------------------------
    # Impression des variables et des groupements
    # -------------------------------------------
    def printVariblesGroupements(self):

            # Exploration de la liste des variables
            print("Liste des variables")
            for variable in self.listeVariablesClass:
                texte = ""
                for valeur in variable.tableau:
                        texte += " " + valeur
                print(("\t" + variable.nom + " :" + texte))

            # Exploration de la liste des groupements
            print("Liste des groupements")
            for groupement in self.listeGroupementsClass:
                print(("\t" + groupement.nom + " (longueur tableau de " + 
                "%.2f" % groupement.lenMin + " a " + 
                str(groupement.lenMax) + ")"))
                for variable in groupement.listeVariables:
                    texte = ""
                    for valeur in variable.tableau:
                        texte += " " + valeur
                    print(("\t\t" + variable.nom + " :" + texte))
    # -------------------------------------------
    # Méthode pour récupérer les blocs de texte depuis la section XML
    def recupBlocsTexteDuXML(self):
        try:
            # Création du minidom
            xmldoc = minidom.parseString(self.texteXML)
            blocsTexteList = xmldoc.getElementsByTagName("blocTexte")
            for bloc in blocsTexteList:
                nomBloc = bloc.getAttribute("nom")
                texteBloc = bloc.childNodes[0].nodeValue
                self.listeBlocsTexte[nomBloc] = texteBloc
        except Exception:
            print("Erreur lors de la récupération des blocs de texte.")
    
    # Méthode pour insérer les blocs de texte dans le contenu
    def insererBlocsTexte(self, texte):
        # Parcourir chaque bloc de texte enregistré
        for nomBloc, contenu in self.listeBlocsTexte.items():
            # Rechercher et remplacer chaque occurrence du marqueur {nomBloc}
            #print(f"Traitement du fichier : {self.nomGeneriqueFichier}")
            #print(f"Remplacement de {{{nomBloc}}} par : {contenu}")
            texte = texte.replace(f"{{{nomBloc}}}", contenu)
        return texte
    # -------------------------------------------
    def printTexteXML(self):
        # Impression de l XML
        print("")
        print((self.texteXML))
        print("")

    # -------------------------------------------
    def printTexte(self):
        print("")
        print((self.texteFichier))
        print("")

#######################################################################
# TRAITEMENT DU JEU DE DONNEES PARAMETRE
#######################################################################

# Recuperation du nom du fichier d entree
nom_fichier = sys.argv[1]

# Affichage du nom
print(("Le nom du fichier d entree est : " + nom_fichier))

# traitement du fichier texte (separation du XML et du texte)
traitementTexte = TraitementTexte(nom_fichier)
traitementTexte.printTexteXML()
#traitementTexte.printTexte()

traitementTexte.recupBlocsTexteDuXML()

# Insérer les blocs de texte dans le texte du fichier avant de récupérer les accolades
traitementTexte.texteFichier = traitementTexte.insererBlocsTexte(traitementTexte.texteFichier)

# Recuperation de variables et des groupements du XML
traitementTexte.recupVarGroupementDuXML()

# Recuperation des accolades
traitementTexte.recuperationAccolades()

# Si la longueur des accolades est differentes stop
isAccoladeGood = True
if len(traitementTexte.ouvertureAccolades) != \
len(traitementTexte.fermetureAccolades) or \
len(traitementTexte.ouvertureAccoladesNom) != \
len(traitementTexte.fermetureAccoladesNom) or \
len(traitementTexte.ouvertureAccolades) == 0:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("Vous avez oublie une accolade pr ouvrir/fermer vos formules")
    print("ou le texte ne contient pas d accolades")
    print("Verifier texte et nom du fichier")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    isAccoladeGood = False

# Recuperation des formules
traitementTexte.recuperationFormules()

# liste fichier Absent
listeFichiersAbsents = deque()

if (len(traitementTexte.listeVariablesClass) > 00 or len(traitementTexte.
listeGroupementsClass) > 0) and isAccoladeGood:

    #--------------------------------------------------------------
    # Creation de la classe listant les variables et les groupements
    # Initialisation du tableauParam a 0, 0, ...
    #--------------------------------------------------------------
    listeDesVariables = \
    ListeDesVariables(traitementTexte.listeVariablesClass,
    traitementTexte.listeGroupementsClass,
    traitementTexte.listeFormulesClass)

    # Impression des limites pour les variables et les groupements
    # print "Limites = "
    # print listeDesVariables.tableauLimites
    # print ""

    #---------------------------------------------------------------------
    # BOUCLE SUR LES VARIABLES ET REMPLACEMENT DES FORMULES
    #---------------------------------------------------------------------
    # Iteration sur les variables et les groupements
    # afin de croiser les parametres
    # Fin de la boucle lorsque on a croise tous les parametres
    while listeDesVariables.tableauParam[0] < \
    listeDesVariables.tableauLimites[0]:
        # Travail sur le n-uplet de parametre
        # chargement des n-uplet dans une liste
        listeDesVariables.chargementUplet()

        # Construction de 3 listes
        #     1 pour les variables calculables
        #     1 pour les variables non calculables
        #     1 troisieme listant toutes les variables
        listeDesVariables.constructionListesVariablesCalculablesOuNon()
        if len(listeDesVariables.isVariablePresenteXfois) > 0:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("Les variables suivantes sont presentes plusieurs fois")
            for variable in listeDesVariables.isVariablePresenteXfois:
                print(variable)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("Echec du PyTiss")
            break

        # Calcul des formules
        traitementTexte.remplacementVariablesFormules(listeDesVariables)
        
        # Générer le nouveau texte sans réinsérer les blocs de texte
        nouveauTexteFichier, nouveauTexteNom = traitementTexte.ecritureNouveauTexte(traitementTexte.texteFichier)
    
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Ecriture du nouveau texte
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Modification du texte avant d'écrire les nouveaux fichiers
        texteAvecBlocs = traitementTexte.insererBlocsTexte(traitementTexte.texteFichier)
        
        nouveauTexteFichier, nouveauTexteNom = \
        traitementTexte.ecritureNouveauTexte(texteAvecBlocs)

        # Impression du nouveau texte
        # print "-----------------------------------------------------"
        # print "              LISTE DES VARIABLES                    "
        # print "-----------------------------------------------------"
        # print listeDesVariables.tableauUplet
        # print "-----------------------------------------------------"
        # print "              IMPRESSION DU FICHIER                  "
        # print "-----------------------------------------------------"
        # print nouveauTexteFichier
        # print "++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        # print "Nom particulier Fichier = " + nouveauTexteNom

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Ecriture du nouveau fichier
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        # Verification presence fichier        
        if os.path.isfile(nouveauTexteNom):
            print(("!!!! Le fichier suivant n a pas ete modifie : " + nouveauTexteNom)) 
        else:
            listeFichiersAbsents.append(nouveauTexteNom)
            nouveauFichier = open(nouveauTexteNom, 'w')
            nouveauFichier.write(nouveauTexteFichier)
            nouveauFichier.close()
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Ecriture de l XML
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        traitementTexte.ecritureSortieXML(listeDesVariables)

        # Passage a l iteration suivante
        traitementTexte.compteurGenerationFichier += 1
        # On passe au n-uplet de parametre suivant
        listeDesVariables.tableauParam[len(listeDesVariables.tableauParam)
        - 1] += 1
        # On verifie que le n-uplet est correct si il est incorrect
        # on corrige
        listeDesVariables.check()

    if len(listeDesVariables.isVariablePresenteXfois) == 0:
        # Ecriture du script de lancement de fichier
        if traitementTexte.scriptLancement != "":
            scriptLancement = open("lancement_" + nom_fichier + ".shell", 'w')
            #texteScript = "#!/bin/sh\r"
            texteScript = ""
            for nomFichier in listeFichiersAbsents:
                texteScript += traitementTexte.scriptLancement \
                + " \"" + nomFichier + "\" ; \r\n"
            scriptLancement.write(texteScript)
            scriptLancement.close()
            subprocess.call('chmod +x lancement_' + nom_fichier + '.shell', shell=True)

        # Ecriture de l XML
        fichierSortieXML = open("sortiePyTissXML" + "_" + nom_fichier + "_" + ".xml", 'w')
        fichierSortieXML.write(traitementTexte.docOutXML.toprettyxml(indent='\t'))
        fichierSortieXML.close()
        
    # Ecriture Bilan Generation
    print(("Nom generique des fichiers : " + traitementTexte.nomGeneriqueFichier))
    print((str(traitementTexte.compteurGenerationFichier) + " fichiers ont ete generes"))

    # Verification de l utilite de creer des dossier
    tableauBool = [False] * len(listeDesVariables.listeUplet[0])

    # boucle sur les uplets
    for iteration in range(0,len(listeDesVariables.listeUplet)-1):

        # Boucle sur le tableau
        for boucle in range(0,len(tableauBool)):
            if tableauBool[boucle] == False :
                if listeDesVariables.listeUplet[iteration][boucle][1] !=  \
                listeDesVariables.listeUplet[iteration+1][boucle][1]:
                    tableauBool[boucle] = True

    # Rangement des fichier
    listeCheminFichier = deque()

    # Boucle sur la liste des chemins de fichier et tableauUplet
    for iteration in range(0,len(traitementTexte.listeNomDeFichier)):
    
        # Creation d une classe pour memoriser le nom du fichier
        # et le chemin associe
        dossier = Dossier()
    
        # Association du nom de fichier
        dossier.nomFichier = traitementTexte.listeNomDeFichier[iteration]
    
        # Association du chemin de fichier
        for boucle in range(0, len(listeDesVariables.listeUplet[iteration])):  
            if tableauBool[boucle]:
                dossier.cheminFichier += listeDesVariables.listeUplet[iteration][boucle][0] + "_" + \
                listeDesVariables.listeUplet[iteration][boucle][1] + "/"
        listeCheminFichier.append(dossier)

    chemin = os.getcwd()

    # Création des dossiers et des sous-dossiers
    if len(sys.argv) > 2 and sys.argv[2] == "CRIS":
        # Script de lancement des CRIS
        scriptCris = open("scriptCRIS_" + nom_fichier + ".shell", 'w')
        texteScriptCris = ""
        # Création des dossiers et du script de lancement des CRIS
        for dossier in listeCheminFichier:
            # Vérifiez si le fichier existe dans le répertoire courant
            if os.path.isfile(dossier.nomFichier):
                # Le fichier existe, vous pouvez le traiter
                # Création du script
                texteScriptCris += "cd " + chemin + "/" + dossier.cheminFichier + ";" + "\n"
                texteScriptCris += "CRIS " + dossier.nomFichier + ";" + "\n"
                texteScriptCris += "cd " + chemin + ";" + "\n"
    
                # Création des dossiers et déplacement des fichiers
                subprocess.call('mkdir -p ' + dossier.cheminFichier, shell=True)
                subprocess.call('mv ' + dossier.nomFichier + " " + dossier.cheminFichier + "/" + dossier.nomFichier, shell=True)
            else:
                # Le fichier n'existe pas dans le répertoire courant
                print(" Le fichier suivant n'a pas été trouvé ou a déjà été traité : " + dossier.nomFichier)
        # Écriture du fichier script CRIS et transformation en exécutable
        scriptCris.write(texteScriptCris)
        scriptCris.close()
        subprocess.call('chmod +x scriptCRIS_' + nom_fichier + '.shell', shell=True)


else:
	if isAccoladeGood == True:
		if traitementTexte.isXMLfind:
			print( "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
			print( "Abscence de donnees XML dans le fichier")
			print( "Les donnees doivent etre encadrees par le noeud donnees.")
			print( "Definir des sous noeuds : variable, groupement, nom")
			print( "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

			print( "\r\nRappel du XML")
			print(( traitementTexte.printTexteXML()))
		else:
			print( "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
			print( "Abscence de donnees XML dans le fichier")
			print( "Les donnees doivent etre encadrees par le noeud donnees.")
			print( "Definir des sous noeuds : variable, groupement, nom")
			print( "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

