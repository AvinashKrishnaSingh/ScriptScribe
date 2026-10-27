import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';

import 'firebase_options.dart';
import 'theme.dart';
import 'welcome_screen.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(options: DefaultFirebaseOptions.currentPlatform);
  runApp(const ScriptScribeApp());
}

class ScriptScribeApp extends StatelessWidget {
  const ScriptScribeApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'ScriptScribe',
        debugShowCheckedModeBanner: false,
        theme: buildTheme(),
        home: const SplashScreen(),
      );
}
