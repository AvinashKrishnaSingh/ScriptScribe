import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'home_screen.dart';
import 'theme.dart';

const _seenWelcomeKey = 'seen_welcome';

Route<void> _fade(Widget page) => PageRouteBuilder(
      pageBuilder: (_, _, _) => page,
      transitionsBuilder: (_, anim, _, child) =>
          FadeTransition(opacity: anim, child: child),
      transitionDuration: const Duration(milliseconds: 500),
    );

/// Animated logo while Firebase auth warms up; routes to welcome or home.
class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with SingleTickerProviderStateMixin {
  late final AnimationController _c = AnimationController(
      vsync: this, duration: const Duration(milliseconds: 1200))
    ..forward();

  @override
  void initState() {
    super.initState();
    _boot();
  }

  Future<void> _boot() async {
    final minimum = Future.delayed(const Duration(milliseconds: 2200));
    final prefs = await SharedPreferences.getInstance();
    try {
      if (FirebaseAuth.instance.currentUser == null) {
        await FirebaseAuth.instance
            .signInAnonymously()
            .timeout(const Duration(seconds: 8));
      }
    } catch (_) {
      // Offline or auth disabled: the app still works, history just won't sync.
    }
    await minimum;
    if (!mounted) return;
    final seen = prefs.getBool(_seenWelcomeKey) ?? false;
    Navigator.pushReplacement(
        context, _fade(seen ? const HomeScreen() : const WelcomeScreen()));
  }

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final curve = CurvedAnimation(parent: _c, curve: Curves.easeOutBack);
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            FadeTransition(
              opacity: _c,
              child: ScaleTransition(
                  scale: Tween(begin: 0.8, end: 1.0).animate(curve),
                  child: const AppLogo(height: 220)),
            ),
            const SizedBox(height: 32),
            const SizedBox(
              width: 28,
              height: 28,
              child: CircularProgressIndicator(strokeWidth: 3, color: kOrange),
            ),
          ],
        ),
      ),
    );
  }
}

class WelcomeScreen extends StatefulWidget {
  const WelcomeScreen({super.key});

  @override
  State<WelcomeScreen> createState() => _WelcomeScreenState();
}

class _WelcomeScreenState extends State<WelcomeScreen> {
  final _controller = PageController();
  int _page = 0;

  static const _pages = [
    (Icons.document_scanner_rounded, 'Scan any script',
        'Snap a photo or pick an image of Sanskrit or Hindi text. Reading happens on your device.'),
    (Icons.translate_rounded, 'Transliterate instantly',
        'Devanagari is converted into readable IAST Roman script, with proper diacritics.'),
    (Icons.auto_stories_rounded, 'Understand the meaning',
        'Get an English translation and keep every scan in your history.'),
  ];

  Future<void> _finish() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_seenWelcomeKey, true);
    if (!mounted) return;
    Navigator.pushReplacement(context, _fade(const HomeScreen()));
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final last = _page == _pages.length - 1;
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            Align(
              alignment: Alignment.centerRight,
              child: TextButton(
                  onPressed: _finish, child: const Text('Skip')),
            ),
            const AppLogo(height: 130),
            Expanded(
              child: PageView.builder(
                controller: _controller,
                itemCount: _pages.length,
                onPageChanged: (i) => setState(() => _page = i),
                itemBuilder: (_, i) {
                  final (icon, title, body) = _pages[i];
                  return Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 32),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Container(
                          padding: const EdgeInsets.all(24),
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            gradient: const LinearGradient(
                                colors: [kTeal, Color(0xFF2E86A8)]),
                            boxShadow: [
                              BoxShadow(
                                  color: kTeal.withValues(alpha: 0.25),
                                  blurRadius: 24,
                                  offset: const Offset(0, 10)),
                            ],
                          ),
                          child: Icon(icon, size: 48, color: Colors.white),
                        ),
                        const SizedBox(height: 28),
                        Text(title,
                            textAlign: TextAlign.center,
                            style: const TextStyle(
                                fontSize: 26,
                                fontWeight: FontWeight.w800,
                                color: kTeal)),
                        const SizedBox(height: 12),
                        Text(body,
                            textAlign: TextAlign.center,
                            style: TextStyle(
                                fontSize: 16,
                                height: 1.5,
                                color: Colors.black.withValues(alpha: 0.6))),
                      ],
                    ),
                  );
                },
              ),
            ),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                for (var i = 0; i < _pages.length; i++)
                  AnimatedContainer(
                    duration: const Duration(milliseconds: 250),
                    margin: const EdgeInsets.all(4),
                    width: i == _page ? 24 : 8,
                    height: 8,
                    decoration: BoxDecoration(
                      color: i == _page ? kOrange : Colors.black12,
                      borderRadius: BorderRadius.circular(8),
                    ),
                  ),
              ],
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(24, 24, 24, 24),
              child: FilledButton(
                onPressed: last
                    ? _finish
                    : () => _controller.nextPage(
                        duration: const Duration(milliseconds: 350),
                        curve: Curves.easeOut),
                child: Text(last ? 'Get started' : 'Next'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
