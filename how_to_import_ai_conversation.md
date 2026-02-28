# Import instruction

## Manuel d'export de données OpenAI

1. Connectez-vous à ChatGPT.
2. Ouvrez `Settings` > `Data Controls`.
3. Cliquez sur `Export data` puis confirmez la demande.
4. Attendez l'email d'OpenAI contenant le lien de téléchargement.
5. Téléchargez l'archive `.zip` et décompressez-la sur votre ordinateur.
6. Ouvrez le fichier principal d'historique (souvent `conversations.json`) pour retrouver vos échanges.

## Manuel d'export de données Gemini

Source : https://www.reddit.com/r/GeminiAI/comments/1r92ehg/how_to_download_your_full_gemini_chat_history/

### Étape 1 : Accéder à Google Takeout et tout désélectionner

1. Allez sur `takeout.google.com`.
2. En haut de la liste, cliquez sur `Deselect all` pour éviter d'exporter des données inutiles.

### Étape 2 : Sélectionner la bonne source (le point clé)

1. Faites défiler la page et **ne cochez pas** `Gemini` (cette option exporte seulement les paramètres de vos Gems).
2. Cherchez plutôt `My Activity`.
3. Cochez `My Activity`.
4. Juste en dessous, cliquez sur `All activity data included`.
5. Dans la fenêtre qui s'ouvre, cliquez sur `Deselect all`, puis cochez uniquement `Gemini Apps`.
6. Cliquez sur `OK`.

### Étape 3 : Lancer l'export

1. Descendez en bas de page et cliquez sur `Next step`.
2. Gardez les options par défaut (`Export once`, format `.zip`) puis cliquez sur `Create export`.
3. Attendez l'email de Google indiquant que le fichier est prêt, puis téléchargez-le.

### Étape 4 : Retrouver le texte des conversations

1. Décompressez le fichier `.zip`.
2. Ouvrez le chemin : `Takeout > My Activity > Gemini Apps`.
3. Ignorez les fichiers/dossiers aux noms étranges (hash) : ils correspondent en général aux pièces jointes (images, PDF, vidéos, snippets de code).
4. Ouvrez le fichier principal `My Activity.html`.
5. Il s'ouvre dans votre navigateur comme une page unique contenant l'historique texte de vos conversations.
6. Utilisez `Ctrl+F` pour retrouver rapidement une conversation ou un mot-clé.
