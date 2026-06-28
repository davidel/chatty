#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <linux/landlock.h>
#include <linux/prctl.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

/* Handle older kernel headers by manually defining missing constants */
#ifndef LANDLOCK_ACCESS_FS_READ_FILE
#define LANDLOCK_ACCESS_FS_EXECUTE (1ULL << 0)
#define LANDLOCK_ACCESS_FS_WRITE_FILE (1ULL << 1)
#define LANDLOCK_ACCESS_FS_READ_FILE (1ULL << 2)
#define LANDLOCK_ACCESS_FS_READ_DIR (1ULL << 3)
#define LANDLOCK_ACCESS_FS_REMOVE_DIR (1ULL << 4)
#define LANDLOCK_ACCESS_FS_REMOVE_FILE (1ULL << 5)
#define LANDLOCK_ACCESS_FS_MAKE_CHAR (1ULL << 6)
#define LANDLOCK_ACCESS_FS_MAKE_DIR (1ULL << 7)
#define LANDLOCK_ACCESS_FS_MAKE_REG (1ULL << 8)
#define LANDLOCK_ACCESS_FS_MAKE_SOCK (1ULL << 9)
#define LANDLOCK_ACCESS_FS_MAKE_FIFO (1ULL << 10)
#define LANDLOCK_ACCESS_FS_MAKE_BLOCK (1ULL << 11)
#define LANDLOCK_ACCESS_FS_MAKE_SYM (1ULL << 12)
#endif

#ifndef SYS_landlock_create_ruleset
#define SYS_landlock_create_ruleset 444
#define SYS_landlock_add_rule 445
#define SYS_landlock_restrict_self 446
#endif

static int landlock_create_ruleset(
    const struct landlock_ruleset_attr* const attr, const size_t size,
    const __u32 flags) {
  return syscall(SYS_landlock_create_ruleset, attr, size, flags);
}

static int landlock_add_rule(const int ruleset_fd,
                             const enum landlock_rule_type rule_type,
                             const void* const rule_attr, const __u32 flags) {
  return syscall(SYS_landlock_add_rule, ruleset_fd, rule_type, rule_attr,
                 flags);
}

static int landlock_restrict_self(const int ruleset_fd, const __u32 flags) {
  return syscall(SYS_landlock_restrict_self, ruleset_fd, flags);
}

int main(int argc, char** argv) {
  __u64 ACCESS_RO = LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR |
                    LANDLOCK_ACCESS_FS_EXECUTE;
  __u64 ACCESS_RW = ACCESS_RO | LANDLOCK_ACCESS_FS_WRITE_FILE |
                    LANDLOCK_ACCESS_FS_MAKE_REG | LANDLOCK_ACCESS_FS_MAKE_DIR |
                    LANDLOCK_ACCESS_FS_REMOVE_FILE |
                    LANDLOCK_ACCESS_FS_REMOVE_DIR;

  struct landlock_ruleset_attr ruleset_attr = {
      .handled_access_fs = ACCESS_RW | LANDLOCK_ACCESS_FS_MAKE_SYM |
                           LANDLOCK_ACCESS_FS_MAKE_FIFO,
  };

  int ruleset_fd =
      landlock_create_ruleset(&ruleset_attr, sizeof(ruleset_attr), 0);
  if (ruleset_fd < 0) {
    perror("Failed to create Landlock ruleset (Kernel < 5.13?)");
    return 1;
  }

  int i = 1;
  int cmd_index = 0;

  for (; i < argc; i++) {
    if (strcmp(argv[i], "--ro") == 0) {
      if (++i >= argc) {
        fprintf(stderr, "Missing path for --ro\n");
        return 1;
      }

      int fd = open(argv[i], O_PATH | O_CLOEXEC);
      if (fd < 0) {
        fprintf(stderr, "Failed to open RO path: %s\n", argv[i]);
        return 1;
      }

      struct stat statbuf;
      if (fstat(fd, &statbuf) < 0) {
        fprintf(stderr, "Failed to stat RO path: %s\n", argv[i]);
        close(fd);
        return 1;
      }

      __u64 allowed = ACCESS_RO;
      if (!S_ISDIR(statbuf.st_mode)) {
        allowed &= (LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_EXECUTE);
      }

      struct landlock_path_beneath_attr path_beneath = {
          .allowed_access = allowed, .parent_fd = fd};
      if (landlock_add_rule(ruleset_fd, LANDLOCK_RULE_PATH_BENEATH,
                            &path_beneath, 0)) {
        perror("failed to add RO rule");
        close(fd);
        return 1;
      }
      close(fd);
    } else if (strcmp(argv[i], "--rw") == 0) {
      if (++i >= argc) {
        fprintf(stderr, "Missing path for --rw\n");
        return 1;
      }

      int fd = open(argv[i], O_PATH | O_CLOEXEC);
      if (fd < 0) {
        fprintf(stderr, "Failed to open RW path: %s\n", argv[i]);
        return 1;
      }

      struct stat statbuf;
      if (fstat(fd, &statbuf) < 0) {
        fprintf(stderr, "Failed to stat RW path: %s\n", argv[i]);
        close(fd);
        return 1;
      }

      __u64 allowed = ACCESS_RW;
      if (!S_ISDIR(statbuf.st_mode)) {
        allowed &= (LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_EXECUTE |
                    LANDLOCK_ACCESS_FS_WRITE_FILE);
      }

      struct landlock_path_beneath_attr path_beneath = {
          .allowed_access = allowed, .parent_fd = fd};
      if (landlock_add_rule(ruleset_fd, LANDLOCK_RULE_PATH_BENEATH,
                            &path_beneath, 0)) {
        perror("failed to add RW rule");
        close(fd);
        return 1;
      }
      close(fd);
    } else if (strcmp(argv[i], "--") == 0) {
      cmd_index = i + 1;
      break;
    }
  }

  if (cmd_index == 0 || cmd_index >= argc) {
    fprintf(stderr, "Usage: %s --ro <path> --rw <path> -- <cmd> [args]\n",
            argv[0]);
    return 1;
  }

  if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)) {
    perror("Failed to restrict privileges");
    return 1;
  }
  if (landlock_restrict_self(ruleset_fd, 0)) {
    perror("Failed to enforce Landlock");
    return 1;
  }
  close(ruleset_fd);

  execvp(argv[cmd_index], &argv[cmd_index]);
  perror("execvp failed");
  return 1;
}
