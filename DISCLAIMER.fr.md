[🇬🇧 English](DISCLAIMER.md) | [🇫🇷 Français](DISCLAIMER.fr.md)

# Avertissement

Ce projet est un **outil de recherche et d'analyse**. Ce n'est pas un conseil
financier.

En utilisant ce code, vous acceptez que :

1. **Pas de conseil financier.** L'auteur n'est ni conseiller financier, ni
   courtier, ni professionnel agréé. Les setups détectés, les scores, les
   hypothèses et tout signal produit par ce logiciel sont une sortie
   exploratoire — pas des recommandations.

2. **Aucune garantie.** Les résultats de backtest et de paper trading ne
   prédisent pas les rendements futurs. Les edges passés se détériorent. La
   méthodologie (`docs/RESEARCH_NOTES.fr.md`) est explicite sur la facilité
   avec laquelle une hypothèse peut être sur-ajustée sur des données
   historiques. Lisez-la avant de faire confiance à n'importe quel chiffre
   produit par cet outil.

3. **Le paper trading est le mode par défaut.** Le mode d'exécution est
   configuré dans `.env` via `EXECUTION_MODE` / `TRADING_MODE`. Le trading
   live doit être activé explicitement et requiert des credentials API
   Bitget valides. N'activez pas le trading live sans comprendre la
   stratégie qui produit les signaux, sans l'avoir fait tourner en paper
   pendant une période prolongée, et sans revoir chaque trade.

4. **Clés API.** Ne commitez jamais votre `.env`. Ne partagez jamais vos
   credentials API. Restreignez votre clé API Bitget au minimum de
   permissions nécessaires et désactivez les permissions de retrait sur
   toute clé utilisée par ce logiciel.

5. **Vous êtes responsable de vos fonds.** L'auteur n'accepte aucune
   responsabilité pour toute perte financière, opportunité manquée,
   indisponibilité d'exchange, corruption de données, ou tout autre dommage
   résultant de l'utilisation de ce logiciel.

6. **Usage légal.** Le trading de dérivés est réglementé dans de nombreuses
   juridictions et interdit aux particuliers dans certaines. Il est de votre
   responsabilité de connaître et de respecter les lois locales.

Si l'un des points ci-dessus n'est pas acceptable pour vous, n'utilisez pas
ce logiciel.
