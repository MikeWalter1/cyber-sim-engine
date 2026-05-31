@echo off

pushd "%~dp0\.."
node runtime\play.js ^
  --deck1 decks\DEV-TEST-001.deck ^
  --deck2 decks\DEV-TEST-001.deck ^
  --bot1 ..\cyber-sim-sdk\server-ai-mybot-v2.js ^
  --bot2 ..\cyber-sim-sdk\server-ai-mybot-v2.js ^
  --runcount 5000
popd
pause 