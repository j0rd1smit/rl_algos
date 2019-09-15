import gym


# Return the minimum amount of fish required to make the story work out with n people
def fishing_trip(n):
    if n <= 1:
        return -1
    i = 1
    while not is_solution(i, n):
        i += 1
    return i

def fishing_trip2(n):
    if n <= 1:
        return -1
    return n ** n - n + 1

def is_solution(number, n):
    for _ in range(n):
        if not number % n == 1:
            return False

        number = (number - 1) // n * (n - 1)

    return True


if __name__ == '__main__':
    print(fishing_trip(3), fishing_trip2(3))
    print(fishing_trip(4), fishing_trip2(4))
    print(fishing_trip(5), fishing_trip2(5))
