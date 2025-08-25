#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "libdwarf/dwarf.h"
#include "libdwarf/libdwarf.h"

void check(int ret, Dwarf_Error &err, const char *msg) {
  if (ret == DW_DLV_ERROR) {
    std::cerr << msg << ": " << dwarf_errmsg(err) << std::endl;
    exit(EXIT_FAILURE);
  }
}

void walk_die(Dwarf_Debug dbg, Dwarf_Die die, int level) {
  Dwarf_Error err = nullptr;
  char *name = nullptr;
  Dwarf_Half tag = 0;

  if (dwarf_diename(die, &name, &err) == DW_DLV_OK) {
    if (dwarf_tag(die, &tag, &err) == DW_DLV_OK) { std::cout << std::string(level * 2, ' ') << "tag=0x" << std::hex << tag << " name=" << (name ? name : "<null>") << '\n'; }
  }

  // 遍历子树
  Dwarf_Die child = nullptr;
  if (dwarf_child(die, &child, &err) == DW_DLV_OK) {
    walk_die(dbg, child, level + 1);
    dwarf_dealloc(dbg, child, DW_DLA_DIE);
  }

  // 遍历兄弟
  Dwarf_Die sibling = nullptr;
  if (dwarf_siblingof(dbg, die, &sibling, &err) == DW_DLV_OK) {
    walk_die(dbg, sibling, level);
    dwarf_dealloc(dbg, sibling, DW_DLA_DIE);
  }

  if (name) dwarf_dealloc(dbg, name, DW_DLA_STRING);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <elf_file>\n";
    return 1;
  }

  int fd = open(argv[1], O_RDONLY);
  if (fd < 0) {
    perror("open");
    return 1;
  }

  Dwarf_Debug dbg = nullptr;
  Dwarf_Error err = nullptr;
  int res = dwarf_init(fd, DW_DLC_READ, nullptr, nullptr, &dbg, &err);
  check(res, err, "dwarf_init");

  Dwarf_Unsigned cu_hdr_len, abbr_off, next_cu;
  Dwarf_Half ver, addr_size;
  while ((res = dwarf_next_cu_header(dbg, &cu_hdr_len, &ver, &abbr_off, &addr_size, &next_cu, &err)) == DW_DLV_OK) {
    Dwarf_Die cu_die = nullptr;
    if (dwarf_siblingof(dbg, nullptr, &cu_die, &err) == DW_DLV_OK) {
      walk_die(dbg, cu_die, 0);
      dwarf_dealloc(dbg, cu_die, DW_DLA_DIE);
    }
  }
  check(res, err, "dwarf_next_cu_header");

  dwarf_finish(dbg, &err);
  close(fd);
  return 0;
}
