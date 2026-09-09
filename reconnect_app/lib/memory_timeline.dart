import 'package:flutter/material.dart';
import 'backend_api.dart';

class MemoryTimelineScreen extends StatefulWidget {
  final int patientId;

  const MemoryTimelineScreen({super.key, required this.patientId});

  @override
  State<MemoryTimelineScreen> createState() => _MemoryTimelineScreenState();
}

class _MemoryTimelineScreenState extends State<MemoryTimelineScreen> {
  static const _bgColor = Color(0xFFF8F6F6);
  static const _textColor = Color(0xFF2F3B47);
  static const _primaryColor = Color(0xFF4F6F8F);

  static const _timeFilters = ['All', 'Today', 'Yesterday', 'This Week', 'This Month'];

  List<Map<String, dynamic>> _memories = [];
  bool _isLoading = true;
  String? _errorMessage;

  String _selectedTimeFilter = 'All';
  String _selectedCategory = 'All';
  String _selectedParticipant = 'All';

  @override
  void initState() {
    super.initState();
    _loadMemories();
  }

  Future<void> _loadMemories() async {
    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });
    try {
      final memories = await BackendApi.allMemories();
      // Newest first, by when the memory actually happened (not last edited).
      memories.sort((a, b) {
        final aTime = DateTime.tryParse(a['latest_event_time'] as String? ?? '');
        final bTime = DateTime.tryParse(b['latest_event_time'] as String? ?? '');
        if (aTime == null || bTime == null) return 0;
        return bTime.compareTo(aTime);
      });
      if (!mounted) return;
      setState(() {
        _memories = memories;
        _isLoading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _errorMessage = 'Could not load memories: $e';
        _isLoading = false;
      });
    }
  }

  // ---------------------------------------------------------------
  // Filter option lists, derived from whatever data actually came back.
  // ---------------------------------------------------------------

  List<String> get _categoryOptions {
    final categories = _memories
        .map((m) => m['category'] as String? ?? '')
        .where((c) => c.isNotEmpty)
        .toSet()
        .toList()
      ..sort();
    return ['All', ...categories];
  }

  List<String> get _participantOptions {
    final people = <String>{};
    for (final memory in _memories) {
      final participants = List<String>.from(
        memory['participants'] as List? ?? const [],
      );
      people.addAll(participants);
    }
    final sorted = people.toList()..sort();
    return ['All', ...sorted];
  }

  // ---------------------------------------------------------------
  // Filtering
  // ---------------------------------------------------------------

  bool _matchesTimeFilter(DateTime dt, DateTime now) {
    switch (_selectedTimeFilter) {
      case 'Today':
        return dt.year == now.year && dt.month == now.month && dt.day == now.day;
      case 'Yesterday':
        final yesterday = DateTime(now.year, now.month, now.day)
            .subtract(const Duration(days: 1));
        return dt.year == yesterday.year &&
            dt.month == yesterday.month &&
            dt.day == yesterday.day;
      case 'This Week':
        final startOfWeek = DateTime(now.year, now.month, now.day)
            .subtract(Duration(days: now.weekday - 1)); // Monday
        final endOfWeek = startOfWeek.add(const Duration(days: 7));
        return !dt.isBefore(startOfWeek) && dt.isBefore(endOfWeek);
      case 'This Month':
        return dt.year == now.year && dt.month == now.month;
      case 'All':
      default:
        return true;
    }
  }

  List<Map<String, dynamic>> get _filteredMemories {
    final now = DateTime.now();
    return _memories.where((memory) {
      final dt = DateTime.tryParse(
        memory['latest_event_time'] as String? ?? '',
      )?.toLocal();
      if (dt == null || !_matchesTimeFilter(dt, now)) return false;

      if (_selectedCategory != 'All' &&
          (memory['category'] as String? ?? '') != _selectedCategory) {
        return false;
      }

      if (_selectedParticipant != 'All') {
        final participants = List<String>.from(
          memory['participants'] as List? ?? const [],
        );
        if (!participants.contains(_selectedParticipant)) return false;
      }

      return true;
    }).toList();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _bgColor,
      body: SafeArea(
        child: RefreshIndicator(
          onRefresh: _loadMemories,
          child: ListView(
            padding: const EdgeInsets.all(20),
            children: [
              _buildHeader(context),
              const SizedBox(height: 20),
              _buildFilterBar(),
              const SizedBox(height: 20),
              if (_isLoading)
                const Padding(
                  padding: EdgeInsets.only(top: 40),
                  child: Center(child: CircularProgressIndicator()),
                )
              else if (_errorMessage != null)
                _buildMessage(_errorMessage!, isError: true)
              else if (_filteredMemories.isEmpty)
                _buildMessage('No memories match these filters yet.')
              else
                for (final memory in _filteredMemories) ...[
                  _buildEventCard(memory),
                  const SizedBox(height: 18),
                ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildMessage(String text, {bool isError = false}) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 40),
      child: Center(
        child: Text(
          text,
          textAlign: TextAlign.center,
          style: TextStyle(
            color: isError ? Colors.red.shade700 : Colors.black54,
            fontSize: 14,
          ),
        ),
      ),
    );
  }

  Widget _buildHeader(BuildContext context) {
    return Row(
      children: [
        InkWell(
          onTap: () => Navigator.of(context).pop(),
          borderRadius: BorderRadius.circular(24),
          child: Container(
            width: 44,
            height: 44,
            decoration: const BoxDecoration(
              color: Colors.white,
              shape: BoxShape.circle,
            ),
            child: const Icon(Icons.chevron_left, color: _textColor),
          ),
        ),
        const SizedBox(width: 12),
        const Expanded(
          child: Text(
            'Memory Timeline',
            style: TextStyle(
              fontSize: 26,
              fontWeight: FontWeight.bold,
              color: _textColor,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildFilterBar() {
    // Only show the dropdowns that actually have more than one option.
    final dropdowns = <Widget>[
      Expanded(
        child: _buildFilterDropdown(
          label: 'When',
          options: _timeFilters,
          selected: _selectedTimeFilter,
          onSelected: (value) => setState(() => _selectedTimeFilter = value),
        ),
      ),
    ];

    if (_categoryOptions.length > 1) {
      dropdowns.add(const SizedBox(width: 10));
      dropdowns.add(
        Expanded(
          child: _buildFilterDropdown(
            label: 'Category',
            options: _categoryOptions,
            selected: _selectedCategory,
            onSelected: (value) => setState(() => _selectedCategory = value),
            formatLabel: _formatCategoryLabel,
          ),
        ),
      );
    }

    if (_participantOptions.length > 1) {
      dropdowns.add(const SizedBox(width: 10));
      dropdowns.add(
        Expanded(
          child: _buildFilterDropdown(
            label: 'People',
            options: _participantOptions,
            selected: _selectedParticipant,
            onSelected: (value) => setState(() => _selectedParticipant = value),
          ),
        ),
      );
    }

    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: dropdowns,
    );
  }

  Widget _buildFilterDropdown({
    required String label,
    required List<String> options,
    required String selected,
    required ValueChanged<String> onSelected,
    String Function(String)? formatLabel,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: const TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w600,
            color: Colors.black54,
            letterSpacing: 0.4,
          ),
        ),
        const SizedBox(height: 8),
        Container(
          height: 44,
          padding: const EdgeInsets.symmetric(horizontal: 14),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: Colors.black12),
          ),
          child: DropdownButtonHideUnderline(
            child: DropdownButton<String>(
              value: selected,
              isExpanded: true,
              icon: const Icon(Icons.keyboard_arrow_down,
                  color: _textColor, size: 20),
              borderRadius: BorderRadius.circular(14),
              dropdownColor: Colors.white,
              style: const TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w600,
                color: _textColor,
              ),
              selectedItemBuilder: (context) {
                return options.map((option) {
                  final text = formatLabel != null
                      ? formatLabel(option)
                      : option;
                  return Align(
                    alignment: Alignment.centerLeft,
                    child: Text(
                      text,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                        color: _textColor,
                      ),
                    ),
                  );
                }).toList();
              },
              items: options.map((option) {
                final isSelected = option == selected;
                final text = formatLabel != null
                    ? formatLabel(option)
                    : option;
                return DropdownMenuItem<String>(
                  value: option,
                  child: Text(
                    text,
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight:
                          isSelected ? FontWeight.w700 : FontWeight.w500,
                      color: isSelected ? _primaryColor : _textColor,
                    ),
                  ),
                );
              }).toList(),
              onChanged: (value) {
                if (value != null) onSelected(value);
              },
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildEventCard(Map<String, dynamic> memory) {
    final category = memory['category'] as String? ?? '';
    final topic = memory['topic'] as String?;
    final summary = memory['summary'] as String? ?? '';
    final participants = List<String>.from(
      memory['participants'] as List? ?? const [],
    );
    final importance = memory['importance'] as String? ?? '';
    final dt = DateTime.tryParse(
      memory['latest_event_time'] as String? ?? '',
    )?.toLocal();

    final title = (topic != null && topic.isNotEmpty)
        ? topic
        : _formatCategoryLabel(category);
    final important = importance.toLowerCase() == 'high';
    final style = _categoryStyle(category);

    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.04),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      clipBehavior: Clip.antiAlias,
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  width: 34,
                  height: 34,
                  decoration: BoxDecoration(
                    color: style.color.withValues(alpha: 0.15),
                    shape: BoxShape.circle,
                  ),
                  child: Icon(style.icon, color: style.color, size: 18),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    title,
                    style: const TextStyle(
                      fontSize: 17,
                      fontWeight: FontWeight.bold,
                      color: _textColor,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 4),
            Padding(
              padding: const EdgeInsets.only(left: 44),
              child: Text(
                dt == null ? '' : _formatTimeLabel(dt),
                style: const TextStyle(fontSize: 13, color: Colors.black54),
              ),
            ),
            const SizedBox(height: 10),
            Text(
              summary,
              style: const TextStyle(
                fontSize: 14,
                color: _textColor,
                height: 1.4,
              ),
            ),
            const SizedBox(height: 14),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Expanded(
                  child: Row(
                    children: [
                      const Icon(Icons.people_alt_outlined,
                          size: 16, color: Colors.black54),
                      const SizedBox(width: 6),
                      Expanded(
                        child: Text(
                          participants.isEmpty
                              ? 'Unknown'
                              : participants.join(', '),
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(
                            fontSize: 13,
                            color: Colors.black54,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
                if (important)
                  Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(
                      color: const Color(0xFFFFF1D6),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: const Row(
                      children: [
                        Icon(Icons.star_border,
                            size: 14, color: Color(0xFFB8860B)),
                        SizedBox(width: 4),
                        Text(
                          'Important',
                          style: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w600,
                            color: Color(0xFFB8860B),
                          ),
                        ),
                      ],
                    ),
                  ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  // ---------------------------------------------------------------
  // Display helpers
  // ---------------------------------------------------------------

  String _formatCategoryLabel(String category) {
    if (category.isEmpty || category == 'All') return category;
    return category
        .split(RegExp(r'[_\s]+'))
        .where((w) => w.isNotEmpty)
        .map((w) => w[0].toUpperCase() + w.substring(1))
        .join(' ');
  }

  _CategoryStyle _categoryStyle(String category) {
    switch (category.toLowerCase()) {
      case 'significant_event':
        return const _CategoryStyle(Icons.star, Color(0xFFCF8B4F));
      case 'identity':
        return const _CategoryStyle(Icons.badge_outlined, Color(0xFF4F6F8F));
      case 'relationship':
        return const _CategoryStyle(Icons.favorite, Color(0xFF4F8FBF));
      case 'commitment':
        return const _CategoryStyle(Icons.event_available, Color(0xFFB8860B));
      case 'preference':
        return const _CategoryStyle(Icons.thumb_up_alt_outlined, Color(0xFF7C5CBF));
      case 'routine_update':
        return const _CategoryStyle(Icons.check_circle_outline, Color(0xFF3F9E8F));
      default:
        return const _CategoryStyle(Icons.chat_bubble_outline, Color(0xFF4F6F8F));
    }
  }

  String _formatTimeLabel(DateTime dt) {
    final now = DateTime.now();
    final today = DateTime(now.year, now.month, now.day);
    final target = DateTime(dt.year, dt.month, dt.day);
    final dayDiff = today.difference(target).inDays;

    final hour12 = dt.hour % 12 == 0 ? 12 : dt.hour % 12;
    final minute = dt.minute.toString().padLeft(2, '0');
    final period = dt.hour < 12 ? 'AM' : 'PM';
    final timePart = '${hour12.toString().padLeft(2, '0')}:$minute $period';

    if (dayDiff == 0) return 'Today, $timePart';
    if (dayDiff == 1) return 'Yesterday, $timePart';
    if (dayDiff > 1 && dayDiff < 7) {
      const weekdays = [
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
      ];
      return '${weekdays[dt.weekday - 1]}, $timePart';
    }
    const months = [
      'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ];
    return '${months[dt.month - 1]} ${dt.day}, $timePart';
  }
}

class _CategoryStyle {
  final IconData icon;
  final Color color;
  const _CategoryStyle(this.icon, this.color);
}