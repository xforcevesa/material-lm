def run_tests():
    from test import decoder_test, mlp_regressor_test, ridge_test
    decoder_test()
    mlp_regressor_test()
    ridge_test()


def main():
    run_tests()


if __name__ == '__main__':
    main()
