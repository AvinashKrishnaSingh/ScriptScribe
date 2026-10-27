import 'package:flutter/material.dart';

const kTeal = Color(0xFF1F5F7A);
const kOrange = Color(0xFFE8913A);
const kBackground = Color(0xFFF8F7F3);

ThemeData buildTheme() {
  final scheme = ColorScheme.fromSeed(
    seedColor: kTeal,
    primary: kTeal,
    secondary: kOrange,
    surface: Colors.white,
  );
  return ThemeData(
    useMaterial3: true,
    colorScheme: scheme,
    scaffoldBackgroundColor: kBackground,
    appBarTheme: const AppBarTheme(
      backgroundColor: kBackground,
      surfaceTintColor: Colors.transparent,
      elevation: 0,
      centerTitle: true,
      titleTextStyle:
          TextStyle(color: kTeal, fontSize: 20, fontWeight: FontWeight.w700),
    ),
    cardTheme: CardThemeData(
      color: Colors.white,
      elevation: 0,
      margin: EdgeInsets.zero,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(20),
        side: BorderSide(color: kTeal.withValues(alpha: 0.08)),
      ),
    ),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(
        minimumSize: const Size.fromHeight(54),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
      ),
    ),
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        minimumSize: const Size.fromHeight(54),
        side: const BorderSide(color: kTeal),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
      ),
    ),
  );
}

class AppLogo extends StatelessWidget {
  final double height;
  const AppLogo({super.key, this.height = 160});

  @override
  Widget build(BuildContext context) =>
      Image.asset('assets/logo.jpg', height: height, fit: BoxFit.contain);
}
