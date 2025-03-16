using Drumwise.Application.Common.Errors;

namespace Drumwise.Application.Common.Models;

public class Result
{
    public bool IsSuccess { get; }
    public bool IsFailure => !IsSuccess;
    public ResultType ResultType { get; set; }
    public IEnumerable<Error> Errors { get; }
    
    private Result(bool isSuccess, IEnumerable<Error> errors, ResultType resultType)
    {
        if (isSuccess && !Equals(errors, Error.None) ||
            !isSuccess && Equals(errors, Error.None))
        {
            throw new ArgumentException("Invalid error", nameof(errors));
        }

        IsSuccess = isSuccess;
        Errors = errors;
        ResultType = resultType;
    }

    public static Result Success(ResultType resultType) => new(true, Error.None, resultType);

    public static Result Failure(IEnumerable<Error> errors, ResultType resultType) => 
        new(false, errors, resultType);
}