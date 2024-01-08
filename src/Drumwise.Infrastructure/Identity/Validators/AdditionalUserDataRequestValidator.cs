using System.Reflection;
using Drumwise.Application.Common.Errors;
using Drumwise.Application.Common.Models.Identity;
using Drumwise.Infrastructure.Identity.Constants;
using FluentValidation;

namespace Drumwise.Infrastructure.Identity.Validators;

public class AdditionalUserDataRequestValidator : AbstractValidator<AdditionalUserDataRequest>
{
    public AdditionalUserDataRequestValidator()
    {
        RuleFor(x => x.Experience)
            .GreaterThanOrEqualTo(0)
            .WithMessage(Error.GetErrorMessage(ErrorCodes.Identity.ExperienceBelowZero))
            .WithErrorCode(ErrorCodes.Identity.ExperienceBelowZero);

        RuleFor(x => x.Name)
            .NotEmpty()
            .WithMessage(Error.GetErrorMessage(ErrorCodes.Identity.NameIsRequired))
            .WithErrorCode(ErrorCodes.Identity.NameIsRequired);
        
        RuleFor(x => x.Surname)
            .NotEmpty()
            .WithMessage(Error.GetErrorMessage(ErrorCodes.Identity.SurnameIsRequired))
            .WithErrorCode(ErrorCodes.Identity.SurnameIsRequired);

        RuleFor(x => x.Role)
            .NotEmpty()
            .WithMessage(Error.GetErrorMessage(ErrorCodes.Identity.RoleIsRequired))
            .WithErrorCode(ErrorCodes.Identity.RoleIsRequired);

        RuleFor(x => x.Role)
            .Must(BeValidRole)
            .WithMessage(Error.GetErrorMessage(ErrorCodes.Identity.UnknownRole))
            .WithErrorCode(ErrorCodes.Identity.UnknownRole);
    }
    
    private static bool BeValidRole(string role)
    {
        if (string.IsNullOrEmpty(role))
            return true;
        
        var fields = typeof(Roles)
            .GetFields(BindingFlags.Public | BindingFlags.Static | BindingFlags.FlattenHierarchy);

        return fields.Any(field => (string)field.GetValue(null)! == role);
    }
}